"""
What a camera's seed left behind, view by view.

The per camera seed (``AbstractTarget.initial_calibration``) poses every view
it solves from and measures what each one leaves, and the intrinsics report
(:mod:`pyCamSet.utils.intrinsics_report`) breaks those numbers down by camera
and by image.  This carries them from the one to the other, so the report
restates the seed's own residuals rather than working them out again.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeedResiduals:
    """
    Each view a camera was seeded from, and the residual it was left with.

    A view is one board in one image for a perspective seed, which poses every
    face of the target on its own, and one image for a telecentric seed,
    which poses the target whole.

    :param rms: each view's RMS reprojection error, in pixels
    :param points: how many points each view holds
    :param image: the image each view came from
    :param has_detection: per image, whether the camera detected anything in it
    """
    rms: np.ndarray
    points: np.ndarray
    image: np.ndarray
    has_detection: np.ndarray

    # The education mode (and older callers) read a camera's seed costs as a plain sequence of
    # per-view RMS values; keep that reading working next to the report's fields.
    def __iter__(self):
        return iter(self.rms)

    def __len__(self) -> int:
        return int(np.size(self.rms))

    def __getitem__(self, index):
        return self.rms[index]
