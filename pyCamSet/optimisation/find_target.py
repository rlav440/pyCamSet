import numpy as np

from pyCamSet import CameraSet, Camera       
from pyCamSet.calibration_targets import AbstractTarget, TargetDetection

from pyCamSet.optimisation.base_optimiser import run_bundle_adjustment
from pyCamSet.optimisation.derived_handlers import StandardBundleParameters

def find_target_pose_at_timestep(
        images: dict[str | int, np.ndarray],
        target: AbstractTarget,
        cameras: CameraSet,
    ):
    """
    Bundle adjustment based optimisation of target position.
    
    :param images: a dictionairy of camera names and an associated image as a numpy array.
    :param target: a calibration target.
    :param cameras: a cameraset that images the target in those image sequences.
    """

    for cam_name in images.keys():
        if not cam_name in cameras.get_names():
            raise ValueError(f"Image from {cam_name}, when {cam_name} not part of the given CameraSet")

    detection = TargetDetection(cam_names=cameras.get_names())
    for cam_name, image in images.values():
        datum = target.find_in_image(image, camera=cameras[cam_name])
        detection.add_detection(detection=datum, cam_name=cam_name, im_num=0)

    fp = {
        c.name:{"ext":c.extrinsic,"int":c.intrinsic,"dst":c.distortion_coefs} 
                for c in cameras
    }

    bundler = StandardBundleParameters(
        cameras, target, detection,
        fixed_params = fp,
        options={"verbosity":0}
    )

    o, _ = run_bundle_adjustment(
        bundler
    )

    _, poses = bundler.get_camset(o.x, return_pose=True)
    return poses[0]


def find_target_poses(
        image_seq:dict[str | int, list[np.ndarray]],
        target: AbstractTarget,
        cameras: CameraSet,
    ):

    for cam_name in image_seq.keys():
        if not cam_name in cameras.get_names():
            raise ValueError("Image from {cam_name}, when {cam_name} not part of the given CameraSet")

    detection = TargetDetection(cam_names=cameras.get_names())
    for cam_name, image_list in image_seq.values():
        for id_im, im in image_list: 
            datum = target.find_in_image(im, camera=cameras[cam_name])
            detection.add_detection(detection=datum, cam_name=cam_name, im_num=id_im)

    fp = {
        c.name:{"ext":c.extrinsic,"int":c.intrinsic,"dst":c.distortion_coefs} 
                for c in cameras
    }

    bundler = StandardBundleParameters(
        cameras, target, detection,
        fixed_params = fp,
        options={"verbosity":0}
    )

    o, _ = run_bundle_adjustment(
        bundler
    )

    _, poses = bundler.get_camset(o.x, return_pose=True)
    return poses
