"""Setting up a bundle adjustment by hand, rather than via calibrate_cameras.

This is a shape-of-the-API sketch, not a runnable script: the camera set and
the detections below are empty placeholders.  In a real calibration `init_cams`
comes from the per-camera intrinsic solve and `detected_data` from running the
target's find_in_image over the calibration images -- see base_calibration.py
for a version that runs end to end against the test data.
"""
from pyCamSet import ChArUco, CameraSet
from pyCamSet.calibration_targets import TargetDetection
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment
from pyCamSet.optimisation.template_handler import TemplateBundleHandler

cam_names = ['cam_0', 'cam_1', 'cam_2']

# The standard initialisation for a camera set is to find the intrinsic
# parameters of each camera.
init_cams = CameraSet()
calibration_target = ChArUco(num_squares_x=10, num_squares_y=10, square_size=4)
detected_data = TargetDetection(cam_names=cam_names)

# Any implementation of this base class can be initialised here.
param_handler = TemplateBundleHandler(
    camset=init_cams,
    target=calibration_target,
    detection=detected_data,
)

# Perform the bundle adjustment based optimisation.
optimisation, optimised_cameras = run_bundle_adjustment(
    param_handler=param_handler,
)
