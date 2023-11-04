import numpy as np
import logging

from pyCamSet import CameraSet  
from pyCamSet.calibration_targets.abstractTarget import AbstractTarget
from pyCamSet.calibration_targets.targetDetections import TargetDetection
from pyCamSet.optimisation.base_optimiser import StandardParamHandler
from pyCamSet.optimisation.compiled_helpers import n_e4x4_flat_INPLACE, n_htform_broadcast_prealloc

from pyCamSet.utils.general_utils import ext_4x4_to_rod


class TwoTargetCalibrator(StandardParamHandler):
    """
    Defines a new class that can calibrate with two! targets that are rigidly fixed together
    """

    def __init__(self, camset: CameraSet, target0: AbstractTarget, target1: AbstractTarget,
                 detection0: TargetDetection, detection1: TargetDetection, 
                 fixed_params: dict|None = None, options: dict|None = None,
                 missing_poses0=None, missing_poses1=None):
        super().__init__(camset, target0, detection0, fixed_params, options, missing_poses0) #use the super to initiate pose, extr, dst and intr automation.

        self.relative_pose = np.eye(4)
        self.extra_detection = detection1
        self.extra_target: AbstractTarget = target1
        self.extra_point_data: np.ndarray = target1.point_data.reshape(-1,3)
        self.extra_missing = missing_poses1
        #unravel the point data beforehand to allow direct concatenation
        self.len0 = target0.point_data.reshape((-1,3)).shape[0] 

        self.point_data = np.concatenate(
            [target0.point_data.reshape((-1, 3)), target1.point_data.reshape((-1,3))], axis=0
        )

    def add_extra_params(self, param_array:np.ndarray) -> np.ndarray:
        """
        Adds 6 extra parameters representing the relative transformation between the two calibration targets used.
        """
        
        base_poses, _ = self.target.pose_in_detections(
            self.detection, self.camset)

        extra_poses, _ = self.extra_target.pose_in_detections(
            self.extra_detection, self.camset)

        estimated_tforms = [np.linalg.inv(b) @ e for b, e in zip(base_poses, extra_poses)]
        average_tform = estimated_tforms[0] #probably fine to take a single instance here
        # the better method would be the average via quaternions.
        return np.concatenate([ext_4x4_to_rod(average_tform), param_array])
    
    def parse_extra_params_and_setup(self, param_array:np.ndarray) -> np.ndarray:
        """
        Takes the first 6 components of the param array, and generates additional points.
        """
        l_t0 = 1
        tform = np.empty(12)
        n_e4x4_flat_INPLACE(param_array[:6], tform)
        n_htform_broadcast_prealloc(self.extra_point_data, tform, self.point_data[l_t0:])
        return param_array[6:]

    def get_detection_data(self, flatten=False):
        if flatten==False:
            raise ValueError("data must be flattened for multiple targets")
        
        data0 = super().get_detection_data(flatten)
        dims = self.extra_target.point_data.shape[:-1]
 
        detection = self.extra_detection
        if self.extra_missing is not None:
            if np.any(self.extra_missing):
                logging.info("Missing poses required removing detected data from the optimisation")
                # delete any inds with missing pose numbers.
                missing_poses = np.where(self.extra_missing)[0]
                detection = self.extra_detection.delete_row(im_num=missing_poses)

        data1 = detection.return_flattened_keys(dims).get_data() + [0,0, self.len0, 0, 0]
        
        return np.concatenate([data0, data1], axis=0)
