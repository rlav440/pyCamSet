"""
A textured height field seen by telecentric cameras, rendered by ray casting.

Shared by the telecentric stereo tests, so it imports nothing test-only.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from pyCamSet.cameras.telecentric_camera import TelecentricCamera
from pyCamSet.utils.general_utils import h_tform

RES = [640, 480]
MAG = 30000.0  # px per metre, a 21 x 16 mm field
BUMP = 0.0015


def make_cam(name, rotvec, k=0.0, eps=0.0, pp_shift=(0.0, 0.0)):
    intrinsic = np.array([[MAG, 0.0, RES[0] / 2 + pp_shift[0]],
                          [0.0, MAG * 1.01, RES[1] / 2 + pp_shift[1]],
                          [0.0, 0.0, 1.0]])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    # a telecentric pose can sit anywhere along its axis; put it well back
    extrinsic[2, 3] = 0.2
    return TelecentricCamera(extrinsic=extrinsic, intrinsic=intrinsic, res=RES,
                             distortion_coefs=np.array([k]), telecentricity=eps, name=name)


def height(x, y):
    return BUMP * np.sin(x / 0.003) * np.cos(y / 0.004)


def texture(x, y):
    """A texture busy at every scale the matcher could care about."""
    rng = np.random.default_rng(3)
    f = rng.uniform(200, 3000, size=(24, 2)) * rng.choice([-1, 1], size=(24, 2))
    ph = rng.uniform(0, 2 * np.pi, 24)
    val = sum(np.sin(f[i, 0] * x + f[i, 1] * y + ph[i]) for i in range(24))
    return np.clip(128 + 20 * val, 0, 255)


def surface_hit(cam, uv, iters=30):
    """Where each pixel's ray meets the height field, by fixed point on the ray."""
    und = cam.undistort_points(uv)
    near = h_tform(cam._pixels_to_cam_frame(und, depth=0.0), cam.cam_to_world)
    far = h_tform(cam._pixels_to_cam_frame(und, depth=1.0), cam.cam_to_world)
    direction = far - near
    t = -near[:, 2] / direction[:, 2]
    for _ in range(iters):
        p = near + t[:, None] * direction
        t = (height(p[:, 0], p[:, 1]) - near[:, 2]) / direction[:, 2]
    return near + t[:, None] * direction


def render(cam):
    u, v = np.meshgrid(np.arange(cam.res[0], dtype=float), np.arange(cam.res[1], dtype=float))
    p = surface_hit(cam, np.stack([u.ravel(), v.ravel()], axis=-1))
    return texture(p[:, 0], p[:, 1]).reshape(u.shape).astype(np.uint8)


def oracle_matcher(rect):
    """A matcher that knows the surface: the true disparity at every left pixel."""
    def matcher(left, right):
        w, h = rect.size
        u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        uv = rect.rectified_to_source(np.stack([u.ravel(), v.ravel()], axis=-1), 'left')
        return rect.expected_disparity(surface_hit(rect.left, uv)).reshape(h, w)
    return matcher
