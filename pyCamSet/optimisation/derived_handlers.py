from __future__ import annotations
from .base_optimiser import AbstractParamHandler

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyCamSet.cameras import CameraSet, Camera
    from pyCamSet.calibration_targets import AbstractTarget, TargetDetection


class StandardBundleParameters(AbstractParamHandler):
    """
    A thin wrapper around the abstract class, which implements all the functionality
    needed for a standard bundle adjustment
    """

    def __init__(self, camset: CameraSet, target: AbstractTarget,
                 detection: TargetDetection, fixed_params: dict|None=None,
                 options: dict | None = None):
        super().__init__(camset, target, detection,
                         fixed_params, options)
