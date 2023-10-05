import numpy as np

from pyCamSet import CameraSet  
from pyCamSet.calibration_targets.abstractTarget import AbstractTarget
from pyCamSet.calibration_targets.targetDetections import TargetDetection
from pyCamSet.optimisation.base_optimiser import AbstractParamHandler
from pyCamSet.optimisation.compiled_helpers import n_e4x4, n_htform_broadcast_prealloc

from pyCamSet.utils.general_utils import ext_4x4_to_rod


class CustomCalibrator(AbstractParamHandler):
    """
    Defines a new class that can calibrate with two! targets that are rigidly fixed together
    """

    def __init__(self, camset: CameraSet, target0: AbstractTarget, target1: AbstractTarget, detection0: TargetDetection, detection1: TargetDetection, fixed_params: dict = None, options: dict = None, missing_poses=None):
        super().__init__(camset, target0, detection0, fixed_params, options, missing_poses) #use the super to initiate pose, extr, dst and intr automation.

        self.relative_pose = np.eye(4)
        self.extra_detection = detection1
        self.extra_target = target1
        self.extra_point_data = target1.point_data
        #unravel the point data beforehand to allow direct concatenation
        self.point_data = np.concatenate(

        )

    def add_extra_params(self, param_array:np.ndarray) -> np.ndarray:
        """
        Adds 6 extra parameters representing the relative transformation between the two calibration targets used.
        """
        
        base_poses, _ = self.target.pose_in_detections(
            self.detection, self.camset)

        extra_poses, _ = self.extra_target.pose_in_detections(
            self.extra_detection, self.camset)

        estimated_tforms = [
            np.linalg.inv(b) @ e for b, e in zip(base_poses, extra_poses)
        ]
        average_tform = quarternion_tform_average(exstimated_tforms) #left as an excersize for the reader.
        return np.concatenate([ext_4x4_to_rod(average_tform), param_array])
    
    def parse_extra_params_and_setup(self, param_array:np.ndarray) -> np.ndarray:
        """
        Takes the first 6 components of the param array, and generates additional points."""
        l_t0 = 1
        tform = n_e4x4(param_array[:6])
        n_htform_broadcast_prealloc(tform, self.extra_target.point_data, self.point_data[l_t0:])
        return param_array[6:]

