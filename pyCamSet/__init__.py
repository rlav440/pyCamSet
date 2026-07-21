from .cameras import CameraSet, Camera
from .utils.saving import load_CameraSet
from .calibration import calibrate_cameras

# Make the native Cairo library discoverable before any target module imports
# `cairosvg`.  This runs once per process and is a no-op when cairo already
# loads (e.g. GUI mode, or a non-conda environment where it is on the PATH).
from .utils.cairo_dll_helper import ensure_cairo_dll_available  # noqa: E402

ensure_cairo_dll_available()

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

try:
    from .calibration_targets.target_puzzleboard import PuzzleBoard
except Exception:
    # PuzzleBoard generation shares the same optional Cairo/native-graphics dependency.
    PuzzleBoard = None  # type: ignore[assignment,misc]

try:
    from .calibration_targets.target_puzzleboard_cube import PuzzleBoardCube
except Exception:
    # PuzzleBoardCube generation shares the same optional Cairo/native-graphics dependency.
    PuzzleBoardCube = None  # type: ignore[assignment,misc]

