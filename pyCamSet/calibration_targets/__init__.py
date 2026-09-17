"""
The things a camera set is calibrated against.

:mod:`core` holds what every target is made of -- the contract, the
detections, the parameters, the registry -- and :mod:`markers` the detectors
that read fiducials out of an image.  Each remaining subpackage is one
target: the class itself in ``target``, and the script that draws one to
paper in ``generate``.

The core names are imported here.  A target class is fetched on first use
instead, through :mod:`core.target_registry`, so that asking for a ChArUco
board does not also import PuzzleBoard's optional detector::

    from pyCamSet.calibration_targets import ChArUco
"""
from .core import AbstractTarget, TargetDetection, ImageDetection, FaceToShape
from .core.target_registry import TARGET_NAMES, target_class

# The target names are written out rather than spliced in from
# TARGET_NAMES.  __all__ is read by tools that parse this file without
# running it -- griffe, which renders the API documentation, is one, and it
# cannot evaluate a name, so under `mkdocs build --strict` an unevaluable
# entry is a failed build.  test_target_names_are_all_exported keeps the two
# lists in step, so a target added to TARGET_CLASSES and not to here fails a
# test rather than quietly going unexported.
__all__ = [
    "AbstractTarget",
    "TargetDetection",
    "ImageDetection",
    "FaceToShape",
    "Ccube",
    "Ccube2",
    "ChArUco",
    "ChArUco2",
    "PuzzleBoard",
    "PuzzleBoardCube",
]


def __getattr__(name: str):
    if name in TARGET_NAMES:
        return target_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
