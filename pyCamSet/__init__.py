import importlib
import logging
from typing import TYPE_CHECKING, Any

# Never true at runtime -- the point of the module is not to import these.
# It is here for the readers that only read: mkdocstrings' griffe resolves
# a lazy name by searching for it, and lands on calibration_targets, whose
# __all__ splices in a list it cannot evaluate, which --strict makes fatal.
# Type checkers and editors follow the same declarations.
if TYPE_CHECKING:
    from .calibration import calibrate_cameras
    from .calibration_targets.ccube.target import Ccube
    from .calibration_targets.charuco.target import ChArUco
    from .calibration_targets.charuco2.target import ChArUco2
    from .calibration_targets.puzzleboard.target import PuzzleBoard
    from .calibration_targets.puzzleboard_cube.target import PuzzleBoardCube
    from .cameras import Camera, CameraSet
    from .utils.calibration_report import CalibrationReport
    from .utils.logs import setup_logging
    from .utils.saving import load_CameraSet

# A library must not decide where its records go. Every pyCamSet module logs
# to a child of this logger; calibrate_cameras installs a coloured handler on
# it when nothing else has configured logging, and setup_logging is there for
# anyone who wants to choose. See pyCamSet.utils.logs.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# The names this package promises, resolved lazily (PEP 562) below rather
# than imported here. `pyCamSet.cameras`, `pyCamSet.calibration` and the
# target modules pull in numba/scipy/etc. on their own, and most callers of
# `import pyCamSet` never touch most of these names -- eagerly importing all
# of them cost every caller a multi-second import even when, say, all they
# wanted was to build one target out of process. See viewer_process.py.
__all__ = [
    "CameraSet",
    "Camera",
    "load_CameraSet",
    "setup_logging",
    "CalibrationReport",
    "calibrate_cameras",
    "ChArUco",
    "ChArUco2",
    "Ccube",
    "PuzzleBoard",
    "PuzzleBoardCube",
]

# Plain re-exports: name -> (module to import, attribute on it). Each is a
# straight `from <module> import <attr>`, just deferred to first access.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "CameraSet": (".cameras", "CameraSet"),
    "Camera": (".cameras", "Camera"),
    "load_CameraSet": (".utils.saving", "load_CameraSet"),
    "setup_logging": (".utils.logs", "setup_logging"),
    "CalibrationReport": (".utils.calibration_report", "CalibrationReport"),
    "calibrate_cameras": (".calibration", "calibrate_cameras"),
}


class _MissingPuzzleBoard:
    """Placeholder raised when the optional PuzzleBoard dependency is absent."""

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "PuzzleBoard requires the optional 'puzzle_board' dependency, which is "
            "not installed. It is not on PyPI, so install it from source: "
            "pip install 'puzzle_board @ "
            "git+https://github.com/PStelldinger/PuzzleBoard.git'."
        )


class _MissingPuzzleBoardCube:
    """Placeholder raised when the optional PuzzleBoard dependency is absent."""

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "PuzzleBoardCube requires the optional 'puzzle_board' dependency, which is "
            "not installed. It is not on PyPI, so install it from source: "
            "pip install 'puzzle_board @ "
            "git+https://github.com/PStelldinger/PuzzleBoard.git'."
        )


def _resolve_charuco() -> Any:
    try:
        from .calibration_targets.charuco.target import ChArUco
    except Exception:
        # cairosvg requires the Cairo native library, which may not be present
        # in all environments. ChArUco generation is optional for reconstruction.
        return None
    return ChArUco


def _resolve_charuco2() -> Any:
    try:
        from .calibration_targets.charuco2.target import ChArUco2
    except Exception:
        # ChArUco2 has no aruco1 equivalent -- it cannot be built at all
        # without aruco2, which may not be installed in all environments.
        return None
    return ChArUco2


def _resolve_ccube() -> Any:
    try:
        from .calibration_targets.ccube.target import Ccube
    except Exception:
        # Ccube generation has the same optional Cairo/native-graphics dependency.
        return None
    return Ccube


def _resolve_puzzleboard() -> Any:
    try:
        from .calibration_targets.puzzleboard.target import PuzzleBoard
    except ModuleNotFoundError as _e:
        if (_e.name or "").split(".")[0] == "puzzle_board":
            return _MissingPuzzleBoard
        return None
    except Exception:
        # PuzzleBoard generation shares the same optional Cairo/native-graphics dependency.
        return None
    return PuzzleBoard


def _resolve_puzzleboard_cube() -> Any:
    try:
        from .calibration_targets.puzzleboard_cube.target import PuzzleBoardCube
    except ModuleNotFoundError as _e:
        if (_e.name or "").split(".")[0] == "puzzle_board":
            return _MissingPuzzleBoardCube
        return None
    except Exception:
        # PuzzleBoardCube generation shares the same optional Cairo/native-graphics dependency.
        return None
    return PuzzleBoardCube


# Each optional target has its own three-way import outcome (class,
# `_Missing...` placeholder, or `None`); see the resolvers above.
_LAZY_RESOLVERS: dict[str, Any] = {
    "ChArUco": _resolve_charuco,
    "ChArUco2": _resolve_charuco2,
    "Ccube": _resolve_ccube,
    "PuzzleBoard": _resolve_puzzleboard,
    "PuzzleBoardCube": _resolve_puzzleboard_cube,
}


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access and cache it in the module globals."""
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
    elif name in _LAZY_RESOLVERS:
        value = _LAZY_RESOLVERS[name]()
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # Cache: later access is a plain attribute lookup.
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
