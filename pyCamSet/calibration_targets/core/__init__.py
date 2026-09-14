"""
What every calibration target is made of.

The contract a target answers to, the detections it hands back, the
parameters it is described by, the net a polyhedral one is folded from, and
the registry that builds one from its name.  Nothing here knows about
markers or about any particular target.
"""
from .parameters import (
    Parameter,
    Parameterisation,
    DocumentedParameters,
    DetectorParameterisation,
)
from .target_detections import TargetDetection, ImageDetection
from .shape_by_faces import FaceToShape
from .abstract_target import AbstractTarget

__all__ = [
    "AbstractTarget",
    "TargetDetection",
    "ImageDetection",
    "FaceToShape",
    "Parameter",
    "Parameterisation",
    "DocumentedParameters",
    "DetectorParameterisation",
]
