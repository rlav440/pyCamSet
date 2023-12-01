
import cv2
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import reduce
import re
import os


from pyCamSet.cameras import CameraSet, Camera
from pyCamSet.calibration_targets import TargetDetection, AbstractTarget
from pyCamSet.optimisation.base_optimiser import run_bundle_adjustment, StandardParamHandler
from pyCamSet.utils.saving import save_pickle, load_pickle, load_CameraSet
from pyCamSet.utils.general_utils import average_tforms, get_subfolder_names, glob_ims, mad_outlier_detection

import coloredlogs, logging

coloredlogs.install(level=logging.INFO)


def calibrate_cameras(
    f_loc: Path|str,
    calibration_target: AbstractTarget,
    save: bool = True,
    save_loc: Path|None = None,
    draw=False,
    n_lim=None,
    fixed_params: dict | None =None,
    high_distortion=False,
    threads=max(1, cpu_count()-2),
    problem_options: dict|None = None,
    ) -> CameraSet:
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
    if isinstance(f_loc, str):
        f_loc = Path(f_loc)

    if save_loc is None:
        save_loc = f_loc


    detections, camera_res = detect_datapoints_in_imfile(
        f_loc=f_loc,
        caching=save,
        calibration_target=calibration_target,
        draw=draw,
        n_lim=n_lim,
    )

    validate_detections(detections, calibration_target)

    string_tail = '.camset'

    initial_cams = run_initial_calibration(
        detections,
        calibration_target,
        camera_res,
        save=save,
        save_loc=save_loc / ('initial_cameras' + string_tail),
        fixed_params=fixed_params
    )

    if high_distortion:
        detections, _ = detect_datapoints_in_imfile(
            f_loc=f_loc,
            calibration_target=calibration_target,
            draw=draw,
            n_lim=n_lim,
            camset=initial_cams
        )

        initial_cams = run_initial_calibration(
            detections,
            calibration_target,
            camera_res,
            save=save,
            save_loc=save_loc / ('initial_cameras_high_distortion' + string_tail),
            )

        initial_cams.draw_camera_distortions()

    initial_cams.set_resolutions_from_file(floc=f_loc)
    calibrated_cameras = run_stereo_calibration(
        initial_cams,
        detections,
        calibration_target,
        save=save,
        save_loc=save_loc/('optimised_cameras' + string_tail),
        fixed_params=fixed_params,
        threads = threads,
        problem_options = problem_options,
    )

    return calibrated_cameras


def run_initial_calibration(detection: TargetDetection,
                            calibration_target: AbstractTarget,
                            cam_res: list[tuple],
                            save=True, save_loc: Path = Path('initial_estimate.camset'),
                            ref_cam: int|str = 0,
                            fixed_params: dict|None = None) -> CameraSet:
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
        cams = load_CameraSet(save_loc)
        return cams

    
    # define the input structure to the
    # inp = data, target, intial_estimate, camera_res
    c_m = detection.features_per_im_per_cam()
    mask = ~np.any(c_m < 6, axis=1)
    score = np.sum(c_m, axis=1)
    pose_im = np.argmax(score * mask)
    # create a lambda based on the inputs

    logging.info("Pulling calibration method from target")
    work_fn = lambda datum: calibration_target.initial_calibration(
            cam_name=datum[0],
            detection=datum[1],
            res=datum[2],
            pose_im=pose_im,
            fixed_params=fixed_params,
        )
    cam_names = detection.cam_names
    cam_detections = detection.get_cam_list()
    work_data = zip(cam_names, cam_detections, cam_res)
    raw_calibration = [work_fn(datum) for datum in work_data]
    cam_dict = {cam_name: cam for cam_name, cam in zip(cam_names, raw_calibration)}
    cams = CameraSet(camera_dict=cam_dict)

    if save:
        cams.save(save_loc)
    return cams


def outlier_rejection(results, params: StandardParamHandler) -> tuple[TargetDetection | None, bool]:
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

    if per_im_outliers is None:
        return None, False
    logging.info("deleting datum associated with the above outliers")
    data = params.detection
    return data.delete_row(im_num=per_im_outliers), True

def run_stereo_calibration(
    cams: CameraSet,
    detections: TargetDetection,
    target: AbstractTarget,
    param_handler = None,
    save: bool=True,
    save_loc: Path|None = None, 
    fixed_params: dict|None=None,
    floc: Path|None=None,
    threads: int = 1, 
    problem_options: dict|None = None,
) -> CameraSet:
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

    if save_loc is None:
        save_loc = Path('optimised_cameras.camset')

    if param_handler is None:
        param_handler = StandardParamHandler(
            detection=detections, target=target, camset=cams,
            fixed_params=fixed_params,
            options=problem_options,
        )

    optimisation, optimised_cams = run_bundle_adjustment(
        param_handler=param_handler,
        threads = threads,
    )

    # outlier_rejection(optimisation.fun.reshape((-1,2)), param_handler)

    param_handler.camset = optimised_cams
    optimised_cams.set_calibration_history(
        optimisation_results=optimisation,
        param_handler=param_handler,
    )

    if save:
        if floc is not None:
            optimised_cams.set_resolutions_from_file(floc)
        optimised_cams.save(save_loc)
    return optimised_cams


def detect_datapoints_in_imfile(
    f_loc: Path,
    calibration_target: AbstractTarget,
    caching=True,
    cache_name='detected_datapoints.pickle',
    draw=False,
    n_lim=None,
    camset:CameraSet|None = None,
    subfolder_string: str|None = None,
) -> tuple[TargetDetection, list[tuple]]:
    """
    This function organises the detection of the image datapoints in a folder of images.

    :param f_loc: the file location to find the images in
    :param calibration_target: the calibration target to use for the detection
    :param caching:  should the result be cached
    :param cache_name: The name of the cache file
    :param draw: Should the detection be drawn
    :param n_lim: The maximum number of images to use for the detection
    :param camset: Optional, a camera set to use for the detections (for high distortion cameras)
    :param subfolder_string: Optional, the name of an intermediate folder bewtween the camera name folder and the image data. 
    :return: A target detection.
    """

    logging.info('starting image detection')

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
                file if subfolder_string is None else file/subfolder_string,
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

        cam_res = [cv2.imread(str(glob_ims(f_loc/cname)[0])).shape[:2] for cname in cam_names]

        # name the detection
        if caching:
            save_pickle((detected, cam_res), f_loc / cache_name)
    else:
        logging.info('loading cached detection')
        detected, cam_res = load_pickle(f_loc / cache_name)
    return detected, cam_res

def validate_detections(detected:TargetDetection, target:AbstractTarget):
    """
    This function checks the detections for each camera, and prints a warning if the detection is poor.
    """
    n_detected = {}

    board_fraction = {}

    corners_per_face = target.point_data.shape[-2]
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


def sanitise_input_images(detected_sub_folders:list[Path], optmode:str='na'):

    """
    Takes a list of detected sub folders and checks that they all have the same number of images.
    :param detected_sub_folders: A list of detected subfolders in the current location.
    :return:
    """
    equal_ims = [len(glob_ims(fol)) for fol in detected_sub_folders]
    if not len(set(equal_ims)) <= 1:
        raise ValueError("An unequal number of calibration images were passed in the input folders.")
