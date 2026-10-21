"""
The cameras of a synthetic capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimCamera:
    """
    One simulated camera.

    :param name: the camera's folder name in the capture.
    :param intrinsic: the 3x3 camera matrix, in OpenCV's convention -- pixel
        centres at integer coordinates.
    :param distortion: OpenCV's five coefficients ``k1, k2, p1, p2, k3``.
    :param res: ``(width, height)`` in pixels.
    :param extrinsic: the 4x4 world-to-camera transform, as
        :class:`~pyCamSet.cameras.Camera` holds it.
    """

    name: str
    intrinsic: np.ndarray
    distortion: np.ndarray
    res: tuple[int, int]
    extrinsic: np.ndarray = field(default_factory=lambda: np.eye(4))

    @property
    def centre(self) -> np.ndarray:
        """Where the camera is, in world coordinates."""
        return np.linalg.inv(self.extrinsic)[:3, 3]

    def to_camera(self):
        """The same camera as a :class:`~pyCamSet.cameras.Camera`."""
        from pyCamSet.cameras import Camera

        return Camera(extrinsic=self.extrinsic.copy(), intrinsic=self.intrinsic.copy(),
                      res=list(self.res), distortion_coefs=self.distortion.copy(),
                      name=self.name)

    def to_dict(self) -> dict:
        return {"name": self.name, "intrinsic": self.intrinsic.tolist(),
                "distortion": self.distortion.tolist(), "res": list(self.res),
                "extrinsic": self.extrinsic.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "SimCamera":
        return cls(name=data["name"], intrinsic=np.asarray(data["intrinsic"], float),
                   distortion=np.asarray(data["distortion"], float),
                   res=tuple(int(v) for v in data["res"]),
                   extrinsic=np.asarray(data["extrinsic"], float))


def look_at(centre, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """
    The world-to-camera transform of a camera at ``centre`` looking at
    ``target``, image x to the right and image y down.

    :param centre: the camera's position.
    :param target: the point it looks at.
    :param up: the world direction that should point up in the image.
    """
    centre = np.asarray(centre, dtype=float)
    z = np.asarray(target, dtype=float) - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=float))
    if np.linalg.norm(x) < 1e-9:                      # looking straight along up
        x = np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ centre
    return extrinsic


def ring_rig(n_cameras: int = 4, radius: float = 0.35, height: float = 0.1,
             res: tuple[int, int] = (1600, 1200), focal: float = 1800.0,
             distortion=(-0.08, 0.03, 0.0, 0.0, 0.0), spread: float = 0.03,
             seed: int = 0) -> list[SimCamera]:
    """
    Cameras on a ring round the origin, all looking at it -- the inward
    facing rig a cube target is made for.

    Each camera is a little different, as real units of one model are: its
    focal length, principal point and distortion are drawn within ``spread``
    (a fraction) of the nominal values, and alternate cameras sit above and
    below the ring by ``height``.

    :param n_cameras: how many cameras.
    :param radius: the ring's radius, in metres.
    :param height: how far above (and, alternately, below) the ring's plane a
        camera sits, in metres.
    :param res: ``(width, height)`` in pixels.
    :param focal: the nominal focal length, in pixels.
    :param distortion: the nominal ``k1, k2, p1, p2, k3``.
    :param spread: how much one camera differs from the next, as a fraction.
    :param seed: the random seed, so a rig can be made again.
    """
    rng = np.random.default_rng(seed)
    width, height_px = res
    cameras = []
    for index in range(n_cameras):
        angle = 2.0 * np.pi * index / n_cameras
        lift = height if index % 2 == 0 else -height
        centre = np.array([radius * np.cos(angle), radius * np.sin(angle), lift])
        f = focal * (1.0 + spread * rng.uniform(-1, 1))
        intrinsic = np.array([
            [f * (1.0 + 0.002 * rng.uniform(-1, 1)), 0.0,
             (width - 1) / 2.0 + spread * width * 0.3 * rng.uniform(-1, 1)],
            [0.0, f, (height_px - 1) / 2.0 + spread * height_px * 0.3 * rng.uniform(-1, 1)],
            [0.0, 0.0, 1.0]])
        dist = np.asarray(distortion, dtype=float) * (1.0 + spread * 3 * rng.uniform(-1, 1, 5))
        cameras.append(SimCamera(name=f"cam_{index}", intrinsic=intrinsic,
                                 distortion=dist, res=(int(width), int(height_px)),
                                 extrinsic=look_at(centre)))
    return cameras
