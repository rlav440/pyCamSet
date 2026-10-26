"""
What a real camera does to the image a lens forms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class SensorModel:
    """
    A camera's degradations, applied to an ideal radiance image.

    The defaults are a typical machine-vision camera under ordinary lab
    lighting: a slightly soft lens, mild vignetting, and the shot and read
    noise of a small-pixel sensor exposed to about two thirds of full well.

    :param blur_sigma: the lens and defocus blur, a Gaussian's standard
        deviation in pixels.
    :param vignetting: how much darker the image's corners are than its
        centre, as a fraction (``cos^4``-like falloff).
    :param full_well: electrons at the brightest level; shot noise is Poisson
        in these.
    :param exposure: the fraction of full well that white paper reaches.
    :param read_noise: read noise, in electrons.
    :param gamma: the output encoding exponent; 8-bit images are written as
        ``linear ** (1 / gamma)``.
    :param seed: the random seed for the noise.
    """

    blur_sigma: float = 0.8
    vignetting: float = 0.25
    full_well: float = 10000.0
    exposure: float = 0.7
    read_noise: float = 8.0
    gamma: float = 2.2
    seed: int = 3

    def to_dict(self) -> dict:
        return asdict(self)

    def apply(self, radiance: np.ndarray, intrinsic: np.ndarray, distortion: np.ndarray,
              rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Turn an ideal pinhole radiance image into what the camera records.

        :param radiance: a float image, 0 black to 1 white paper, drawn with
            the pinhole model and ``intrinsic``.
        :param intrinsic: the camera matrix it was drawn with.
        :param distortion: OpenCV's five coefficients, to bend the image by.
        :param rng: the noise generator; one from ``seed`` when not given.
        :return: a uint8 image.
        """
        rng = np.random.default_rng(self.seed) if rng is None else rng
        image = distort_image(radiance, intrinsic, distortion)
        if self.blur_sigma > 0:
            image = cv2.GaussianBlur(image, (0, 0), self.blur_sigma)
        if self.vignetting > 0:
            image = image * vignetting_map(image.shape, intrinsic, self.vignetting)
        electrons = np.clip(image, 0.0, None) * self.exposure * self.full_well
        electrons = rng.poisson(electrons).astype(np.float64)
        electrons += rng.normal(0.0, self.read_noise, electrons.shape)
        linear = np.clip(electrons / self.full_well, 0.0, 1.0)
        return np.round(255.0 * linear ** (1.0 / self.gamma)).astype(np.uint8)


#: The last few cameras' distortion maps, by camera and image size: every
#: image a camera takes is bent the same way, and building the map is most of
#: the cost of bending one.
_MAPS: dict = {}
_MAPS_HELD = 8


def distort_image(image: np.ndarray, intrinsic: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """
    Bend a pinhole image by a lens's distortion.

    Each output pixel is where a distorted ray lands; the ideal pixel it came
    from is found by undistorting that position, and the image sampled there.

    :param image: the pinhole image.
    :param intrinsic: its camera matrix.
    :param distortion: OpenCV's five coefficients.
    """
    if not np.any(distortion):
        return image
    height, width = image.shape[:2]
    key = (np.asarray(intrinsic, np.float64).tobytes(),
           np.asarray(distortion, np.float64).tobytes(), height, width)
    maps = _MAPS.get(key)
    if maps is None:
        us, vs = np.meshgrid(np.arange(width, dtype=np.float64),
                             np.arange(height, dtype=np.float64))
        ideal_x, ideal_y = undistort_normalised(
            (us - intrinsic[0, 2]) / intrinsic[0, 0], (vs - intrinsic[1, 2]) / intrinsic[1, 1],
            np.asarray(distortion, dtype=np.float64))
        maps = ((ideal_x * intrinsic[0, 0] + intrinsic[0, 2]).astype(np.float32),
                (ideal_y * intrinsic[1, 1] + intrinsic[1, 2]).astype(np.float32))
        if len(_MAPS) >= _MAPS_HELD:
            _MAPS.pop(next(iter(_MAPS)))
        _MAPS[key] = maps
    return cv2.remap(image.astype(np.float32), maps[0], maps[1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def undistort_normalised(xd, yd, distortion, iterations: int = 30):
    """
    Invert OpenCV's five-coefficient distortion on normalised coordinates,
    by fixed-point iteration -- what ``cv2.undistortPoints`` does, run long
    enough to converge to well under a thousandth of a pixel at the corners.
    """
    k1, k2, p1, p2, k3 = (list(distortion) + [0.0] * 5)[:5]
    x, y = xd.copy(), yd.copy()
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


def vignetting_map(shape, intrinsic: np.ndarray, strength: float) -> np.ndarray:
    """A ``cos^4``-shaped falloff scaled so the corners lose ``strength``."""
    height, width = shape[:2]
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    x = (us - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (vs - intrinsic[1, 2]) / intrinsic[1, 1]
    falloff = 1.0 / (1.0 + x ** 2 + y ** 2) ** 2
    corner = falloff.min()
    return 1.0 - strength * (1.0 - falloff) / max(1.0 - corner, 1e-12)
