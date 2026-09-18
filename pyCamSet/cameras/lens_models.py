"""
The lens models a calibration can be asked for, by name.

A camera's lens model decides which camera class is built, which function
blocks the bundle adjustment compiles, and -- for a telecentric lens -- whether
the fit is conditioned at all: a pinhole model fitted to telecentric imagery
drives the focal length towards infinity and leaves the residual in the
distortion coefficients.

The names live here rather than beside the camera classes so that an interface
can offer them without importing the optimisation machinery, and so that the
calibrator and the interface cannot drift apart about what is accepted.
"""
from __future__ import annotations

#: the name of the model assumed when a caller does not say
DEFAULT_LENS_MODEL = "pinhole"

#: every model :func:`pyCamSet.calibrate_cameras` will fit, in offering order
LENS_MODELS: tuple[str, ...] = ("pinhole", "telecentric")

#: what an interface shows for each name
LENS_MODEL_LABELS: dict[str, str] = {
    "pinhole": "Pinhole",
    "telecentric": "Telecentric",
}

#: why a user would pick each one, for a tooltip or a help string
LENS_MODEL_DESCRIPTIONS: dict[str, str] = {
    "pinhole": (
        "A conventional lens, where a point's image moves as it approaches the "
        "camera. The right choice unless the optics are telecentric."
    ),
    "telecentric": (
        "A telecentric lens, whose rays are parallel, so magnification does not "
        "change with distance. Fitting a pinhole model to one of these is badly "
        "conditioned: the focal length runs away and the distortion "
        "coefficients absorb what is left."
    ),
}


def lens_model_label(name: str) -> str:
    """
    What an interface shows for a lens model name.

    :param name: a key of :data:`LENS_MODELS`
    :return: its label, or the name itself when it has none
    """
    return LENS_MODEL_LABELS.get(name, name)
