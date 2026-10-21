from .acmmp_utils import ReconParams
from .reconstruction_utils import (
    depth_image_ptcloud_mask,
    disparity_to_ptcld,
    rectify_camera_images,
    rectify_camera_pair,
    remap_im,
    stereo_reconstruct,
    undistort_im,
)
# telecentric_stereo imports the camera classes, and the camera module imports
# this package through utils.saving, so these resolve on first use
_TELECENTRIC = {
    "TelecentricRectification",
    "reconstruct_telecentric_pair", "rectify_telecentric_pair", "triangulate_pair",
}


def __getattr__(name):
    if name in _TELECENTRIC:
        from . import telecentric_stereo
        return getattr(telecentric_stereo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
