from pyCamSet.optimisation.base_optimiser import run_bundle_adjustment, TemplateBundleHandler

from pyCamSet import ChArUco, CameraSet
from pyCamSet.calibration_targets import TargetDetection


#creating the initial data. For a calibration, these should be populated.

# The standard initialisation for a camera set is to find the intrinsic parameters of each camera
init_cams = CameraSet()
calibration_target = ChArUco()
detected_data = TargetDetection()

# any implementation of this base class can be initialised here.
param_handler = TemplateBundleHandler(
    camset=init_cams,
    target=calibration_target,
    detection=detected_data,
)

# perform the bundle adjustment based optimisation.
optimisation, optimised_cameras = run_bundle_adjustment(
    param_handler=param_handler,
)


