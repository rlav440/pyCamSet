"""
Synthetic calibration captures: a known camera rig photographing a known,
slightly imperfect target, written out as an image folder the real pipeline
reads.

The point is a ground truth.  A real capture says how well a calibration
agrees with itself; a synthetic one also says how far it is from the cameras
and the target that actually made the images.  Each module is one part of
that imaging chain, so a part can be swapped without touching the others:

- :mod:`.rig` -- the cameras: intrinsics, distortion, where they stand.
- :mod:`.poses` -- where the target is held for each shot.
- :mod:`.imperfection` -- how a printed net sits on a real solid: each face
  pasted a little turned, shifted and stretched.
- :mod:`.render` -- the target's faces drawn into each camera's image.
- :mod:`.sensor` -- what a real camera does to that image: lens distortion,
  blur, vignetting, shot and read noise, 8-bit quantisation.
- :mod:`.capture` -- all of it, written to ``<folder>/<camera>/<shot>.png``
  with the ground truth beside it.
- :mod:`.evaluate` -- a calibration compared against that ground truth.

See ``docs/how-to/simulate.md`` for a walk-through, and
``scripts/simulate_calibration.py`` for the end-to-end driver.
"""

from .capture import Capture, load_ground_truth, simulate_capture
from .evaluate import compare_to_truth
from .imperfection import PastingErrors, paste_faces
from .poses import held_target_poses
from .rig import SimCamera, ring_rig
from .sensor import SensorModel

__all__ = [
    "Capture",
    "PastingErrors",
    "SensorModel",
    "SimCamera",
    "compare_to_truth",
    "held_target_poses",
    "load_ground_truth",
    "paste_faces",
    "ring_rig",
    "simulate_capture",
]
