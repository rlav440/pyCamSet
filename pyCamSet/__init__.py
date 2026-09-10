import logging

# A library must not decide where its records go. Every pyCamSet module logs
# to a child of this logger; calibrate_cameras installs a coloured handler on
# it when nothing else has configured logging, and setup_logging is there for
# anyone who wants to choose. See pyCamSet.utils.logs.
logging.getLogger(__name__).addHandler(logging.NullHandler())

from .cameras import CameraSet, Camera
from .utils.saving import load_CameraSet
from .utils.logs import setup_logging
from .utils.calibration_report import CalibrationReport
from .calibration import calibrate_cameras
from .calibration_targets.target_charuco import ChArUco
from .calibration_targets.target_Ccube import Ccube
