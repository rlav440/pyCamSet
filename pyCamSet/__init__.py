from .cameras import CameraSet, Camera
from .utils.saving import load_CameraSet
from .calibration import calibrate_cameras

try:
    from .calibration_targets.target_charuco import ChArUco
except Exception:
    # cairosvg requires the Cairo native library, which may not be present
    # in all environments. ChArUco generation is optional for reconstruction.
    ChArUco = None  # type: ignore[assignment,misc]

try:
    from .calibration_targets.target_Ccube import Ccube
except Exception:
    # Ccube generation has the same optional Cairo/native-graphics dependency.
    Ccube = None  # type: ignore[assignment,misc]

from .calibration_targets.target_puzzleboard import PuzzleBoard
from .calibration_targets.target_puzzleboard_cube import PuzzleBoardCube

