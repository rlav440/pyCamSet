"""Bundle adjustment based pose estimation for a known target in a known rig.

The other use of a calibration target: once the cameras are calibrated, the
target is a well characterised fiducial.  Here the cameras and the target
geometry are held fixed and only the target's pose per image is solved for.

This module was written against ``pyCamSet.optimisation.base_optimiser`` and
``derived_handlers``, both of which were removed, so it had not been importable
for some time.  It is now built on ``optimisation_handling.run_bundle_adjustment``
and ``TemplateBundleHandler`` -- the same pair ``calibrate_cameras`` uses.
"""

from __future__ import annotations

import numpy as np

from pyCamSet.calibration_targets import TargetDetection
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment
from pyCamSet.optimisation.template_handler import TemplateBundleHandler
from pyCamSet.utils.general_utils import ext_4x4_to_rod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyCamSet.calibration_targets import AbstractTarget
    from pyCamSet.cameras import CameraSet


def fix_all_cameras(cameras: CameraSet) -> dict:
    """
    Builds the fixed_params dict that holds every camera at its calibration.

    The handler writes these values straight into its parameter arrays, so
    they have to be in the packed layout those arrays use rather than as
    matrices: six numbers for an extrinsic (rodrigues rotation then
    translation) and nine for an intrinsic (fx, cx, fy, cy, then the five
    distortion coefficients).  Distortion lives inside ``int``; there is no
    separate key for it.

    :param cameras: the calibrated camera set to hold fixed
    :return: a fixed_params dict, keyed by camera name
    """
    fixed = {}
    for cam in cameras:
        rot, trans = ext_4x4_to_rod(cam.extrinsic)
        k = cam.intrinsic
        fixed[cam.name] = {
            "ext": np.concatenate([rot, trans]),
            "int": np.concatenate(
                [[k[0, 0], k[0, 2], k[1, 1], k[1, 2]], np.reshape(cam.distortion_coefs, -1)]
            ),
        }
    return fixed


def _poses_to_4x4(flat_poses: np.ndarray) -> np.ndarray:
    """Turns the handler's (n, 12) flat poses into (n, 4, 4) transforms.

    The flat layout is the rotation's nine elements followed by the three
    translation components -- see n_e4x4_flat_INPLACE -- not a row-major 3x4,
    so the two blocks have to be unpacked separately.
    """
    flat_poses = np.reshape(flat_poses, (-1, 12))
    n = len(flat_poses)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = np.reshape(flat_poses[:, :9], (n, 3, 3))
    poses[:, :3, 3] = flat_poses[:, 9:]
    return poses


def _check_cameras_cover(names, cameras: CameraSet) -> None:
    """Every named camera must be part of the set, or the detection is bogus."""
    known = cameras.get_names()
    for cam_name in names:
        if cam_name not in known:
            raise ValueError(
                f"Image from {cam_name}, when {cam_name} not part of the given CameraSet"
            )


def _initial_pose_params(
        detection: TargetDetection, target: AbstractTarget, cameras: CameraSet
) -> np.ndarray:
    """Estimates a starting pose per image, packed as the handler wants them.

    The handler's own ``calc_initial_params`` cannot be used here.  It runs
    outlier rejection across the set of target poses, whose spread is
    undefined for a single image, and it re-estimates the camera extrinsics --
    which is precisely what this module holds fixed.  ``pose_in_detections``
    is no good either: it prompts on stdin.

    :return: the free parameters, which with every camera fixed are exactly
        six numbers per image
    """
    params = []
    for im_num, im_detection in enumerate(detection.get_image_list()):
        for cam in cameras:
            # target -> camera, or NaN when this camera cannot see enough of it
            in_cam = target.target_pose_in_cam_image(im_detection, cam, mode="nan")
            if not np.any(np.isnan(in_cam)):
                pose = cam.cam_to_world @ in_cam  # target -> world
                rot, trans = ext_4x4_to_rod(pose)
                params.append(np.concatenate([rot, trans]))
                break
        else:
            raise ValueError(
                f"Could not estimate a target pose for image {im_num} from any "
                f"camera: the target needs at least 8 detected points in one view."
            )
    return np.concatenate(params)


def _solve_poses(
        detection: TargetDetection, target: AbstractTarget, cameras: CameraSet
) -> np.ndarray:
    """Runs the bundle adjustment with every camera held fixed."""
    bundler = TemplateBundleHandler(
        camset=cameras,
        target=target,
        detection=detection,
        fixed_params=fix_all_cameras(cameras),
        # fixed_pose defaults to 0, which is the right gauge choice when
        # calibrating -- the target defines the world frame.  Here the
        # calibrated cameras already define it, and the target pose is the
        # unknown, so fixing one would leave that image with nothing to solve
        # (and, for a single image, nothing to solve at all).
        options={"verbosity": 0, "fixed_pose": []},
    )
    bundler.set_initial_params(_initial_pose_params(detection, target, cameras))

    optimisation, _ = run_bundle_adjustment(bundler)
    _, flat_poses = bundler.get_camset(optimisation.x, return_pose=True)
    return _poses_to_4x4(flat_poses)


def find_target_pose_at_timestep(
        images: dict[str | int, np.ndarray],
        target: AbstractTarget,
        cameras: CameraSet,
    ) -> np.ndarray:
    """
    Bundle adjustment based optimisation of target position.

    :param images: a dictionairy of camera names and an associated image as a numpy array.
    :param target: a calibration target.
    :param cameras: a cameraset that images the target in those image sequences.
    :return: the target's pose as a 4x4 homogenous transform
    """
    _check_cameras_cover(images.keys(), cameras)

    detection = TargetDetection(cam_names=cameras.get_names())
    # .items(), not .values(): the values are the images alone, so unpacking a
    # name out of them either raised or silently shredded the array.
    for cam_name, image in images.items():
        datum = target.find_in_image(image, camera=cameras[cam_name])
        detection.add_detection(detection=datum, cam_name=cam_name, global_im_num=0)

    if not detection.has_data():
        raise ValueError("The target was not detected in any of the given images")

    return _solve_poses(detection, target, cameras)[0]


def find_target_poses(
        image_seq: dict[str | int, list[np.ndarray]],
        target: AbstractTarget,
        cameras: CameraSet,
    ) -> np.ndarray:
    """
    Bundle adjustment based optimisation of a sequence of target positions.

    Every camera's list is indexed by timestep, so entry i of each list must be
    the same instant.

    :param image_seq: a dictionairy of camera names and a list of images.
    :param target: a calibration target.
    :param cameras: a cameraset that images the target in those image sequences.
    :return: the target's poses as an (n_timesteps, 4, 4) array
    """
    _check_cameras_cover(image_seq.keys(), cameras)

    detection = TargetDetection(cam_names=cameras.get_names())
    for cam_name, image_list in image_seq.items():
        # enumerate, not iteration: image_list holds images, so unpacking an
        # index out of each one was never going to work.
        for global_im_num, im in enumerate(image_list):
            datum = target.find_in_image(im, camera=cameras[cam_name])
            detection.add_detection(detection=datum, cam_name=cam_name, global_im_num=global_im_num)

    if not detection.has_data():
        raise ValueError("The target was not detected in any of the given images")

    return _solve_poses(detection, target, cameras)
