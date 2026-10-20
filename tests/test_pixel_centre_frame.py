"""
Every ChArUco-family target returns its corners in the pixel-centre frame
(integer coordinates are pixel centres), whichever OpenCV is installed, and a
corner read from a resized image comes back to the same frame.

The boards are rendered analytically: a texture whose pixel k spans
[k, k + 1) is mapped into the image by a homography H, and each image pixel
(i, j), centred on (i, j), is the mean of a grid of samples around its centre,
each reading the texture at floor(H^-1 p). A texture point u is therefore at
H u in the pixel-centre frame, which is the truth the detections are held to.
OpenCV 4.x's CharucoDetector reports corners half a pixel off that frame and
OpenCV 5's on it, so these pass under both only if the conversion is right.
"""
import cv2
import numpy as np
import pytest

from pyCamSet.calibration_targets.charuco import ChArUco
from pyCamSet.calibration_targets.core.abstract_target import (
    _prepare_detection_image,
    _restore_native_coordinates,
)
from pyCamSet.calibration_targets.markers.aruco2 import ARUCO2_AVAILABLE

SHAPE = (640, 800)          # image rows, columns
TEX_PER_SQUARE = 60         # texture pixels per ChArUco square
SAMPLES = 4                 # samples per image pixel, per axis


def _render(texture: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """*texture* drawn through *homography* (texture corner frame -> image pixel-centre frame)."""
    inverse = np.linalg.inv(homography)
    offsets = (np.arange(SAMPLES) + 0.5) / SAMPLES - 0.5
    xs = (np.arange(SHAPE[1])[:, None] + offsets[None, :]).ravel()
    ys = (np.arange(SHAPE[0])[:, None] + offsets[None, :]).ravel()
    x, y = np.meshgrid(xs, ys)
    w = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2]
    u = np.floor((inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]) / w).astype(np.int64)
    v = np.floor((inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]) / w).astype(np.int64)
    inside = (u >= 0) & (u < texture.shape[1]) & (v >= 0) & (v < texture.shape[0])
    values = np.full(u.shape, 255.0)
    values[inside] = texture[v[inside], u[inside]]
    image = values.reshape(SHAPE[0], SAMPLES, SHAPE[1], SAMPLES).mean(axis=(1, 3))
    image = cv2.GaussianBlur(25.0 + image * (210.0 / 255.0), (0, 0), 0.6)  # off saturation, lightly blurred
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def _homography(texture_shape, angle_deg: float, shift, scale: float) -> np.ndarray:
    th, tw = texture_shape
    a = np.radians(angle_deg)
    centre = np.array([[1, 0, -tw / 2], [0, 1, -th / 2], [0, 0, 1.0]])
    rotate = np.array([[scale * np.cos(a), -scale * np.sin(a), 0],
                       [scale * np.sin(a), scale * np.cos(a), 0], [0, 0, 1.0]])
    tilt = np.array([[1, 0, 0], [0, 1, 0], [4e-5, -3e-5, 1.0]])
    place = np.array([[1, 0, SHAPE[1] / 2 + shift[0]], [0, 1, SHAPE[0] / 2 + shift[1]], [0, 0, 1.0]])
    return place @ tilt @ rotate @ centre


#: (angle, sub-pixel shift): four poses whose corners fall at different sub-pixel phases.
POSES = [(7.0, (0.13, 0.37)), (-9.0, (0.41, -0.22)), (3.0, (-0.29, 0.05)), (-4.0, (0.0, 0.0))]


def _charuco_scenes(target: ChArUco):
    board = target.board
    texture = board.generateImage((9 * TEX_PER_SQUARE, 7 * TEX_PER_SQUARE), marginSize=0, borderBits=1)
    corners_u = (np.asarray(board.getChessboardCorners()).reshape(-1, 3)[:, :2]
                 / target.square_size * TEX_PER_SQUARE)
    for angle, shift in POSES:
        homography = _homography(texture.shape, angle, shift, 1.0)
        truth = cv2.perspectiveTransform(corners_u[None], homography)[0]
        yield _render(texture, homography), truth


def _grid_board_scenes(target):
    from pyCamSet.calibration_targets.markers.aruco2_gridboard import render_grid_board_image
    texture = np.asarray(render_grid_board_image(target.grid_size, target._aruco_dict_int, 3))
    square, band = 12, 3.0  # a 4x4 dictionary at bit size 3: 12 px squares, a 3 px band
    cols, rows = np.meshgrid(np.arange(target.grid_size[0] + 1), np.arange(target.grid_size[1] + 1))
    lattice_u = np.stack([band + cols.ravel() * square, band + rows.ravel() * square], axis=-1)
    for angle, shift in POSES:
        homography = _homography(texture.shape, angle, shift, 4.5)
        yield _render(texture, homography), cv2.perspectiveTransform(lattice_u[None], homography)[0]


#: OpenCV's sub-pixel refinement run to convergence. OpenCV 4.x's CharucoDetector
#: starts cornerSubPix half a pixel back and stops at its default 0.1 px accuracy,
#: which leaves corners up to ~0.02 px short of the saddle on a 30 px square
#: (measured: 0.500 px between 4.11 and 5.0 once converged, 0.48 at the default),
#: and a 0.5x read doubles that on the way back. That is the detector's accuracy,
#: not its frame, so the frame is checked with the refinement converged.
CONVERGED = {"cornerRefinementMinAccuracy": 0.001, "cornerRefinementMaxIterations": 100}


def _targets():
    out = [("ChArUco-aruco1", lambda opts: ChArUco(9, 7, 10.0, 0.6, marker_backend="aruco1",
                                                   detection_options=opts), _charuco_scenes)]
    if ARUCO2_AVAILABLE:
        from pyCamSet.calibration_targets.charuco2 import ChArUco2
        out.append(("ChArUco-aruco2", lambda opts: ChArUco(9, 7, 10.0, 0.6, marker_backend="aruco2"),
                    _charuco_scenes))
        out.append(("ChArUco2", lambda opts: ChArUco2(8, 6, 10.0), _grid_board_scenes))
    return out


def _read(target, image, scale):
    """The target's points in *image*, read at *scale* and mapped back as the detection pass does."""
    prepared, used = _prepare_detection_image(
        image, rescale_and_gamma=scale != 1.0, preprocessing_scale=scale, preprocessing_gamma=1.0)
    return _restore_native_coordinates(target.find_in_image(prepared), used)


def _median_error(name, target, scenes, scale):
    errors = []
    for image, truth in scenes(target):
        detection = _read(target, image, scale)
        assert detection.has_data, f"{name}: nothing read at scale {scale}"
        keys = np.asarray(detection.keys).reshape(len(detection.image_points), -1)[:, -1]
        errors.append(np.asarray(detection.image_points, dtype=np.float64) - truth[keys])
    errors = np.concatenate(errors)
    assert len(errors) > 100, f"{name}: only {len(errors)} corners read at scale {scale}"
    return np.median(errors, axis=0)


@pytest.fixture
def converged_aruco2_refinement(monkeypatch):
    """aruco2's ChArUco adapter with its CharucoDetector's refinement run to convergence."""
    from pyCamSet.calibration_targets.markers import aruco2 as adapter
    from pyCamSet.calibration_targets.markers.aruco_opencv import ARUCO_OPENCV_DETECTOR
    options = dict(ARUCO_OPENCV_DETECTOR.resolve(None), **CONVERGED)
    monkeypatch.setattr(adapter, "_charuco_detector_for",
                        lambda board: ARUCO_OPENCV_DETECTOR.build_detector(board, options))


@pytest.mark.parametrize("scale", [1.0, 0.5, 1.5])
@pytest.mark.parametrize("name, make, scenes", _targets(), ids=lambda v: v if isinstance(v, str) else "")
def test_corners_land_in_the_pixel_centre_frame(name, make, scenes, scale, converged_aruco2_refinement):
    median = _median_error(name, make(CONVERGED), scenes, scale)
    # Half a pixel is the size of the frame error this guards against; 0.02 px is far inside it.
    assert np.all(np.abs(median) < 0.02), (
        f"{name} at scale {scale} (OpenCV {cv2.__version__}): median offset from the pixel-centre truth "
        f"{median.round(4)} px")


@pytest.mark.parametrize("scale", [1.0, 1.5])
@pytest.mark.parametrize("name, make, scenes", _targets(), ids=lambda v: v if isinstance(v, str) else "")
def test_default_settings_land_in_the_pixel_centre_frame(name, make, scenes, scale):
    """The targets as they are built by default, where OpenCV 4.x's early stop costs little (see CONVERGED)."""
    median = _median_error(name, make(None), scenes, scale)
    assert np.all(np.abs(median) < 0.02), (
        f"{name} at scale {scale} (OpenCV {cv2.__version__}): median offset from the pixel-centre truth "
        f"{median.round(4)} px")
