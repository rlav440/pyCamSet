from functools import reduce
from multiprocessing import cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pyCamSet.cameras import CameraSet
from pyCamSet.calibration_targets import TargetDetection, AbstractTarget
from pyCamSet.calibration.detection_cache import (
    detection_cache_name, load_verified_cache, save_to_cache,
)
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment
from pyCamSet.optimisation.template_handler import (
    TemplateBundleHandler, DEFAULT_OPTIONS)
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
from pyCamSet.utils.paths import long_path
from pyCamSet.utils.saving import load_CameraSet
from pyCamSet.utils.general_utils import get_subfolder_names, glob_ims

import logging

from pyCamSet.utils.logs import setup_logging_from_verbosity
from pyCamSet.utils.intrinsics_report import report_initial_calibration
from pyCamSet.utils.setup_reports import validate_detections

logger = logging.getLogger(__name__)


def calibrate_cameras(
    f_loc: Path|str,
    calibration_target: AbstractTarget,
    save: bool = True,
    save_loc: Path|None = None,
    draw=False,
    n_lim=None,
    fixed_params: dict | None =None,
    high_distortion=False,
    threads=None,
    problem_options: dict|None = None,
    initial_cams: CameraSet | None= None,
    min_detections_per_board: int = 12,
    optimise_target: bool = False,
    model: str = "pinhole",
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
    :param model: the lens model to calibrate, "pinhole" or "telecentric". Every
        camera in a run shares one model, because one kernel is compiled per
        calibration.
    :param min_detections_per_board: Minimum number of detected corners required
        for a board observation to contribute to the initial per-camera calibration.
    :param optimise_target: solve the target's own geometry as well, in a second
        bundle adjustment started from the first.
    """

    setup_logging_from_verbosity(
        (problem_options or {}).get("verbosity", DEFAULT_OPTIONS["verbosity"]))

    f_loc = Path(f_loc)
    save_loc = f_loc if save_loc is None else save_loc
    if threads is None:
        threads = min(max(1, cpu_count()-2), 20) # don't DOS servers

    detections, camera_res = detect_datapoints_in_imfile(
        f_loc=f_loc,
        caching=save,
        calibration_target=calibration_target,
        draw=draw,
        n_lim=n_lim,
        threads=threads,
    )

    validate_detections(detections, calibration_target,
                        image_counts=images_per_camera(f_loc), n_lim=n_lim)

    if initial_cams is None:
        initial_cams = run_initial_calibration(
            detections,
            calibration_target,
            camera_res,
            save=save,
            save_loc=save_loc / 'initial_cameras.camset',
            fixed_params=fixed_params,
            min_detections_per_board=min_detections_per_board,
            model=model,
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
                save_loc=save_loc / 'initial_cameras_high_distortion.camset',
                min_detections_per_board=min_detections_per_board,
                model=model,
                )

            # only open a window when someone is there to close it
            initial_cams.draw_camera_distortions(show=draw)
    else:
        logger.info("Using the provided initial cameras.")

    initial_cams.set_resolutions_from_file(floc=f_loc)
    report_initial_calibration(initial_cams, detections, calibration_target,
                               min_detections_per_board)
    if len(initial_cams) == 1:
        logger.warning("Only found and calibrated one camera - returning single camera calibration")
        return initial_cams

    calibrated_cameras = run_stereo_calibration(
        initial_cams,
        detections,
        calibration_target,
        save=False, #TODO REMOVE
        save_loc=save_loc/'optimised_cameras.camset',
        fixed_params=fixed_params,
        threads = threads,
        problem_options = problem_options,
    )

    if optimise_target:
        calibrated_cameras = run_self_calibration(
            calibrated_cameras,
            detections,
            calibration_target,
            fixed_params=fixed_params,
            threads=threads,
            problem_options=problem_options,
        )

    return calibrated_cameras


def run_self_calibration(
    cams: CameraSet,
    detections: TargetDetection,
    target: AbstractTarget,
    fixed_params: dict|None = None,
    threads: int = 1,
    problem_options: dict|None = None,
) -> CameraSet:
    """
    Solves the target's own geometry, starting from a finished calibration.

    The same observations are solved again with every feature coordinate
    free. Scale is gauged by three of the target's points.

    :param cams: the cameras a fixed target calibration produced
    :param detections: the detections that calibration was solved against
    :param target: the calibration target, as drawn
    :param fixed_params: parameters the optimisation is not allowed to move
    :param threads: evaluation threads for the compiled kernels
    :param problem_options: options passed through to the solver
    :return: the camera set that minimises the free target problem
    """
    logger.info("Running the calibration again with the target's geometry free")
    param_handler = SelfBundleHandler(
        camset=cams, target=target, detection=detections,
        fixed_params=fixed_params, options=problem_options,
    )
    param_handler.set_from_templated_camset(cams)
    _, self_calibrated = run_bundle_adjustment(
        param_handler=param_handler, threads=threads)
    return self_calibrated


def run_initial_calibration(detection: TargetDetection,
                            calibration_target: AbstractTarget,
                            cam_res: list[tuple],
                            save=True, save_loc: Path = Path('initial_estimate.camset'),
                            ref_cam: int|str = 0,
                            fixed_params: dict|None = None,
                            return_poses_and_costs=False,
                            min_detections_per_board: int = 12,
                            model: str = "pinhole") -> CameraSet | tuple[CameraSet, list, list]:
    """
    For all of the cameras, runs the calibration method provided by an abstract target.
    The default is a closed form seed that fits no distortion, leaving that to
    the bundle adjustment, but may be overwritten.

    :param detection: the detection data to use for the calibration
    :param calibration_target: the calibration target to use for the calibration
    :param save: should the result be saved
    :param save_loc: where should the result be saved
    :param fixed_params: a dictionary of fixed parameters for the optimisation, which will not be changed
    :param model: the lens model to fit, "pinhole" or "telecentric"
    :param min_detections_per_board: Minimum number of detected corners required
        for a board observation to contribute to the initial per-camera calibration.
    :return: the camera set with the initial calibration
    """

    if save_loc.exists() and save:
        logger.info(f"Loading a previously saved initial calib from {save_loc}")
        cams = load_CameraSet(save_loc)
        return (cams, [], []) if return_poses_and_costs else cams

    features = detection.features_per_im_per_cam()
    if features.size == 0:
        raise ValueError(
            "No detection features were found for any camera/image. "
            "Check that the calibration target matches the detected board "
            "type and that the test data contains valid images."
        )
    usable = ~np.any(features < 6, axis=1)
    pose_im = np.argmax(np.sum(features, axis=1) * usable)

    logger.info("Pulling calibration method from target")
    work_fn = lambda cam_name, cam_detection, res: calibration_target.initial_calibration(
        cam_name=cam_name,
        detection=cam_detection,
        res=res,
        pose_im=pose_im,
        fixed_params=fixed_params,
        return_poses=return_poses_and_costs,
        min_detections_per_board=min_detections_per_board,
        model=model,
    )
    cam_names = detection.cam_names
    results = [work_fn(*datum) for datum in
               zip(cam_names, detection.get_cam_list(), cam_res)]

    if return_poses_and_costs:
        raw_calibration = [res[0] for res in results]
        poses = [res[1] for res in results]
        per_im = [res[2] for res in results]
    else:
        raw_calibration, poses, per_im = results, [], []

    cams = CameraSet(camera_dict=dict(zip(cam_names, raw_calibration)))
    if save:
        cams.save(save_loc)

    return (cams, poses, per_im) if return_poses_and_costs else cams


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
    :param floc: the location of the images, used to update the camera resolutions.
                 When supplied, set_resolutions_from_file() is always called regardless
                 of the save flag, so that the returned CameraSet has correct resolution
                 data even when save=False.
    """
    logger.info("Running the full multiview calibration")

    if save_loc is None:
        save_loc = Path('optimised_cameras.camset')

    if param_handler is None:
        param_handler = TemplateBundleHandler(
            detection=detections, target=target, camset=cams,
            fixed_params=fixed_params,
            options=problem_options,
        )

    _, optimised_cams = run_bundle_adjustment(
        param_handler=param_handler,
        threads = threads,
    )
    param_handler.camset = optimised_cams

    if floc is not None:
        optimised_cams.set_resolutions_from_file(floc)
    if save:
        optimised_cams.save(save_loc)
    return optimised_cams



def detect_datapoints_in_imfile(
    f_loc: Path,
    calibration_target: AbstractTarget,
    caching=True,
    draw=False,
    n_lim=None,
    camset:CameraSet|None = None,
    subfolder_string: str|None = None,
    threads=1,
    upscale_factor:int=1,
    cam_names: list[str] | None = None,
    rescale_and_gamma: bool = False,
    preprocessing_scale: float = 0.25,
    preprocessing_gamma: float = 0.5,
) -> tuple[TargetDetection, list[tuple]]:
    """
    This function organises the detection of the image datapoints in a folder of images.

    :param f_loc: the file location to find the images in
    :param calibration_target: the calibration target to use for the detection
    :param caching:  should the result be cached
    :param draw: Should the detection be drawn
    :param n_lim: The maximum number of images to use for the detection
    :param camset: Optional, a camera set to use for the detections (for high distortion cameras)
    :param subfolder_string: Optional, the name of an intermediate folder bewtween the camera name folder and the image data.
    :param cam_names: the exact camera folder names this pass must read,
        when the caller already knows them (phase1.py passes a selection
        fixed before any staging decision), so a folder appearing in or
        vanishing from *f_loc* afterwards cannot join or leave the pass.
        ``None`` scans *f_loc* here instead.
    :param rescale_and_gamma: preprocess with a rescale-and-gamma pass
        before detecting (pcube's low-contrast, low-resolution imagery)
        rather than plain upscaling
    :param preprocessing_scale: the rescale factor when *rescale_and_gamma*
        is set
    :param preprocessing_gamma: the gamma correction when *rescale_and_gamma*
        is set
    :return: A target detection.
    """

    logger.info('starting image detection')

    cache_name = detection_cache_name(calibration_target, camset, upscale_factor)

    cache_path = f_loc / cache_name
    scan_cam_names = cam_names is None
    cam_names = get_subfolder_names(f_loc=f_loc) if scan_cam_names else list(cam_names)

    # Folded into the cache identity (not the cache name) so switching this
    # setting invalidates a stale slot instead of silently reading detections
    # made under a different preprocessing pass.
    preprocessing = None
    if rescale_and_gamma:
        preprocessing = {
            "rescale_and_gamma": True,
            "scale": float(preprocessing_scale),
            "gamma": float(preprocessing_gamma),
        }
    cache_hit = (load_verified_cache(
        cache_path, calibration_target, cam_names, n_lim, camset=camset,
        preprocessing=preprocessing)
        if caching else None)
    if cache_hit is not None:
        logger.info('loading cached detection')
        return cache_hit

    if not caching:
        logger.info('Not caching, starting detection')
    elif not long_path(cache_path).exists():
        logger.info('No cached detection, starting detection')
    else:
        logger.info('Cached detection does not match this target, camera '
                    'selection or image cap, or was written by an earlier '
                    'version; redetecting')

    detected_sub_folders = (
        get_subfolder_names(f_loc, return_full_path=True) if scan_cam_names
        else [f_loc / name for name in cam_names])
    if not detected_sub_folders:
        raise ValueError(f'no subfolders were found in {f_loc}')
    sanitise_input_images(detected_sub_folders)

    work_fn = lambda file, cam=None: \
        calibration_target.find_in_imfolder(
            file if subfolder_string is None else file/subfolder_string,
            cam_names=cam_names,
            draw=draw,
            n_lim=n_lim,
            camera=cam,
            threads=threads,
            upscale_factor=upscale_factor,
            rescale_and_gamma=rescale_and_gamma,
            preprocessing_scale=preprocessing_scale,
            preprocessing_gamma=preprocessing_gamma,
        )

    if camset is not None:
        cam_zip = [camset[f.parts[-1]] for f in detected_sub_folders]
        detections = [work_fn(file, cam) for file, cam in zip(tqdm(detected_sub_folders), cam_zip)]
    else:
        detections = [work_fn(file) for file in tqdm(detected_sub_folders)]
    detected = reduce(lambda x, y: x + y, detections)

    # cam_res must reflect the upscaled coordinate frame, not native.
    # When upscale_factor > 1, detected 2D pixel coords are in the upscaled
    # frame, so cam_res must match. Multiply native .shape[:2] by the factor
    # (cheaper than re-reading the image and resizing it). A rescale-and-gamma
    # pass does not change the coordinate frame the way upscaling does.
    coordinate_scale = 1 if rescale_and_gamma else upscale_factor
    cam_res = [tuple(int(d * coordinate_scale) for d in cv2.imread(str(glob_ims(f_loc/cname)[0])).shape[:2]) for cname in cam_names]

    if caching:
        save_to_cache(detected, cam_res, cache_path, calibration_target,
                      cam_names, n_lim, camset, preprocessing=preprocessing)
    return detected, cam_res


def images_per_camera(f_loc: Path) -> dict[str, int]:
    """
    How many images each camera folder under *f_loc* holds.

    The denominator of the detection rate.

    :param f_loc: the folder holding the per camera sub folders
    """
    return {folder.name: len(glob_ims(folder))
            for folder in get_subfolder_names(f_loc, return_full_path=True)}


def sanitise_input_images(detected_sub_folders: list[Path]):
    """
    Takes a list of detected sub folders and checks that they all have the same number of images.

    :param detected_sub_folders: A list of detected subfolders in the current location.
    """
    image_counts = {len(glob_ims(fol)) for fol in detected_sub_folders}
    if len(image_counts) > 1:
        raise ValueError("An unequal number of calibration images were passed in the input folders.")
