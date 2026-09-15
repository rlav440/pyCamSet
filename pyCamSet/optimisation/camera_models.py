"""
Which function blocks a camera's lens model contributes to a bundle adjustment.

A calibration compiles one kernel from one chain of function blocks, so every
camera in a set has to share a lens model.  This is the one place that knows
which blocks each model brings, and the one place that refuses a set which
mixes them.
"""
from __future__ import annotations

from typing import Type

import pyCamSet.optimisation.function_block_implementations as fb
from pyCamSet.cameras.camera import Camera
from pyCamSet.cameras.telecentric_camera import TelecentricCamera

#: camera class -> (intrinsic block, extrinsic block)
MODEL_BLOCKS: dict[type, tuple[Type[fb.abstract_function_block], Type[fb.abstract_function_block]]] = {
    Camera: (fb.projection, fb.extrinsic3D),
    TelecentricCamera: (fb.telecentric_intrinsic, fb.telecentric_extrinsic),
}


def blocks_for_camset(camset) -> tuple[Type[fb.abstract_function_block], Type[fb.abstract_function_block]]:
    """
    The intrinsic and extrinsic blocks for every camera in a set.

    :param camset: the cameras about to be calibrated
    :raises ValueError: if the set mixes lens models, or holds an unknown one
    :return: the intrinsic and extrinsic block classes
    """
    models = {type(cam) for cam in camset}
    if len(models) > 1:
        names = ", ".join(sorted(m.__name__ for m in models))
        raise ValueError(
            "A calibration compiles a single kernel from a single chain of "
            f"function blocks, so every camera in a set has to share a lens "
            f"model. This set holds {names}. Calibrate each model's cameras as "
            "their own set, then register the results."
        )
    model = models.pop()
    if model not in MODEL_BLOCKS:
        raise ValueError(
            f"{model.__name__} has no bundle adjustment blocks. Add it to "
            "MODEL_BLOCKS in pyCamSet.optimisation.camera_models, with an "
            "intrinsic block that projects a camera frame point to pixels and "
            "an extrinsic block that carries the parameters of its pose."
        )
    return MODEL_BLOCKS[model]
