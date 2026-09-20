from __future__ import annotations

import logging

import cv2
import numpy as np

from pyCamSet.cameras.camera import Camera, _require_pyvista
from pyCamSet.utils.general_utils import h_tform, px_array

try:
    import pyvista as pv
    _PYVISTA_OK = True
except ImportError:
    _PYVISTA_OK = False

logger = logging.getLogger(__name__)

# Scales the squared radius so k is in (1000 px)**-2 rather than carrying the
# target's length unit. The compiled telecentric_intrinsic block repeats the
# number as a literal, since a generated kernel cannot import it, and
# test_block_matches_the_camera_model holds the two together.
R2_SCALE = 1e-6

#: fractional magnification error that bounds the drawn working volume
MAGNIFICATION_ERROR_BUDGET = 1e-3
#: how many field widths deep the drawn volume may get before it is capped, so
#: that a near-perfect lens does not produce a box that dwarfs its own rig
VIEW_DEPTH_CAP_FIELDS = 10.0

DEFAULT_TELECENTRIC_RES = [1000, 1000]
DEFAULT_TELECENTRIC_MATRIX = np.array(
    [[20000.0, 0.0, DEFAULT_TELECENTRIC_RES[0] / 2],
     [0.0, 20000.0, DEFAULT_TELECENTRIC_RES[1] / 2],
     [0.0, 0.0, 1.0]])  # magnification in px per world unit


class TelecentricCamera(Camera):
    """
    A camera behind a telecentric lens.

    A telecentric lens puts its aperture stop at the focal point, so the chief
    rays are parallel in object space and magnification is very nearly
    independent of depth.  The projection is therefore affine rather than
    perspective::

        w = 1/(1 + eps*z)
        u = m_x * x * w * d + c_x
        v = m_y * y * w * d + c_y

    where ``d`` is the division-model distortion factor and ``eps`` is the
    residual telecentricity error -- zero for a perfect lens, in which case the
    projection is purely affine and depth drops out entirely.

    Magnification lives in the ``intrinsic`` matrix where a pinhole camera
    keeps its focal length, in pixels per world unit.  ``distortion_coefs``
    holds the single division coefficient.

    Three things follow from the geometry and are worth knowing before reading
    a calibrated telecentric camera's extrinsic:

    - The camera's position along its own optical axis is not observable.  The
      bundle adjustment does not estimate it, and ``extrinsic`` carries
      whatever the seed put there.
    - The in-plane position is carried by the principal point, not the
      translation, because the two are degenerate.
    - Only the rotation of ``extrinsic`` is estimated.  Relative camera
      geometry within a rig is recovered as rotation plus principal point.
    """

    def __init__(self,
                 extrinsic=np.eye(4),
                 intrinsic=None,
                 res=None,
                 distortion_coefs=np.array([0.0]),
                 telecentricity: float = 0.0,
                 name: str = None,
                 minimal=True):
        """
        Initialises a telecentric camera

        :param extrinsic: the camera extrinsic matrix
        :param intrinsic: magnification and principal point, laid out as a pinhole K
        :param res: the camera resolution
        :param distortion_coefs: the single division model coefficient
        :param telecentricity: residual telecentricity error, 0 for a perfect lens
        :param name: the camera name
        :param minimal: lazy generation of the sensor map
        """
        if res is None:
            res = DEFAULT_TELECENTRIC_RES
        if intrinsic is None:
            intrinsic = DEFAULT_TELECENTRIC_MATRIX
        # set before super().__init__, which calls _update_state and so reads them
        self.telecentricity = float(telecentricity)
        self._view_depth: float | None = None
        super().__init__(
            extrinsic=extrinsic, intrinsic=intrinsic, res=res,
            distortion_coefs=np.reshape(np.asarray(distortion_coefs, dtype=float), -1),
            name=name, minimal=minimal,
        )

    # the model itself

    @property
    def magnification(self) -> np.ndarray:
        """The x and y magnifications, in pixels per world unit."""
        return np.array([self.intrinsic[0, 0], self.intrinsic[1, 1]])

    @property
    def principal_point(self) -> np.ndarray:
        """The distortion centre, in pixels."""
        return np.array([self.intrinsic[0, 2], self.intrinsic[1, 2]])

    @property
    def field_size(self) -> np.ndarray:
        """The imaged object extent in world units -- the telecentric analogue of a field of view."""
        return np.asarray(self.res, dtype=float) / self.magnification

    def __eq__(self, other) -> bool:
        if not super().__eq__(other):
            return False
        return bool(np.isclose(self.telecentricity, other.telecentricity))

    def _update_optical_state(self):
        """
        A telecentric lens has no focal point and no angular field of view.

        Both of the pinhole quantities are read out of ``intrinsic[0, 0]``,
        which here is a magnification.  What replaces them is the imaged extent
        in world units, which does not depend on where the camera is.
        """
        self.focal_point = None
        self.fov = None

    def _cam_fov(self):
        raise AttributeError(
            "a telecentric camera has no angular field of view; its imaged "
            "extent in world units is TelecentricCamera.field_size"
        )

    def _calc_projection_matrix(self):
        """
        The exact 3x4 projection matrix of the undistorted lens.

        ``u = m*x/(1 + eps*z) + c`` is projective, not just affine: it is the
        matrix below divided by its own last row.  A perfect lens leaves that
        row ``[0, 0, 0, 1]``, so the divide is by one and the rays meet at
        infinity; a real one puts the centre at ``z = -1/eps``.  Either way
        triangulation's DLT needs no special case.

        :return matrix: the projection matrix of the camera
        """
        m_x, m_y = self.magnification
        c_x, c_y = self.principal_point
        eps = self.telecentricity
        projective = np.array([
            [m_x, 0.0, c_x * eps, c_x],
            [0.0, m_y, c_y * eps, c_y],
            [0.0, 0.0, eps, 1.0],
        ])
        return projective @ self.extrinsic

    def _distort(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Applies the division model to pixel offsets from the principal point."""
        k = float(np.reshape(self.distortion_coefs, -1)[0])
        if k == 0.0:
            return xs, ys
        den = 1.0 / (1.0 + k * (xs * xs + ys * ys) * R2_SCALE)
        return xs * den, ys * den

    def _undistort(self, xd: np.ndarray, yd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Inverts :meth:`_distort` in closed form.

        Writing ``d`` for the distortion factor, the forward model gives
        ``rd2 = r2*d**2`` and ``d*(1 + k*r2) = 1``, so ``d`` solves
        ``d**2 - d + k*rd2 = 0``.  The root that tends to one at the centre is
        the physical one.  Real telecentric lenses distort by well under a
        percent, so the discriminant only goes negative for coefficients far
        outside anything a calibration would produce -- but it is clipped
        rather than left to produce NaNs.
        """
        k = float(np.reshape(self.distortion_coefs, -1)[0])
        if k == 0.0:
            return xd, yd
        rd2 = (xd * xd + yd * yd) * R2_SCALE
        disc = 1.0 - 4.0 * k * rd2
        if np.any(disc < 0):
            logger.warning(
                "%s: %d points lie outside the radius this division model can "
                "invert (k=%.4g); they are clamped to its limit",
                self.name, int(np.sum(disc < 0)), k,
            )
        den = 0.5 * (1.0 + np.sqrt(np.clip(disc, 0.0, None)))
        return xd / den, yd / den

    def project_points(self, points, mode="opencv", distort=True):
        """
        Projects a list of points onto camera coordinates

        :param points: The points to project
        :param mode: by default, returns in opencv coordinates.
                    with mode == opencv
                    alternate method is mode == "image"
                    which returns v,u coordinates
        :return points: points in the uv coordinates
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, ...]
        cam_pts = h_tform(points, self.extrinsic)
        if cam_pts.ndim == 1:
            cam_pts = cam_pts[None, ...]

        w = 1.0 / (1.0 + self.telecentricity * cam_pts[:, 2])
        m_x, m_y = self.magnification
        c_x, c_y = self.principal_point
        xs = m_x * cam_pts[:, 0] * w
        ys = m_y * cam_pts[:, 1] * w

        if distort:
            xs, ys = self._distort(xs, ys)

        uv = np.stack([xs + c_x, ys + c_y], axis=-1)
        if mode == "image":
            return uv[:, ::-1]
        return uv

    def undistort_points(self, uv: np.ndarray) -> np.ndarray:
        """
        Maps detected pixels back to where an undistorted lens would have put them.

        :param uv: an (n, 2) array of detected pixel coordinates
        :return: the same points with this camera's distortion removed
        """
        uv = np.asarray(uv, dtype=float)
        if uv.ndim == 1:
            uv = uv[None, ...]
        centre = self.principal_point
        xs, ys = self._undistort(uv[:, 0] - centre[0], uv[:, 1] - centre[1])
        return np.stack([xs + centre[0], ys + centre[1]], axis=-1)

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Removes this camera's distortion from an image.

        ``cv2.undistort`` cannot be used: it assumes a pinhole camera with a
        Brown-Conrady model.  The closed-form inverse lets this be a single
        remap with no iteration.

        :param image: An input image
        :return: An undistorted image
        """
        cols, rows = np.meshgrid(
            np.arange(self.res[0], dtype=float), np.arange(self.res[1], dtype=float))
        c_x, c_y = self.principal_point
        # the map sends each output pixel to the source pixel it draws from, so
        # it is the forward distortion, not the inverse
        xs, ys = self._distort(cols - c_x, rows - c_y)
        return cv2.remap(
            image, (xs + c_x).astype(np.float32), (ys + c_y).astype(np.float32),
            cv2.INTER_LINEAR)

    # rays and sensor maps

    def _make_sensormap(self, mode='linear', distort=True):
        """
        Builds the camera's map of pixel to camera frame direction.

        Every ray is parallel to the optical axis, so what varies across the
        sensor is where the ray starts, not where it points.  The map is still
        the camera frame point at depth one behind each pixel, which is what
        ``_compute_world_sensor_map`` and ``im_to_world_ray`` expect.

        :param mode: normalised or linear sensor map
        :param distort: whether to undistort the pixel grid first
        """
        if mode not in ('linear', 'normalised'):
            raise ValueError("Invalid sensor map type")
        u, v, _ = px_array(res=self.res, startZero=True)
        uv = np.stack([u.ravel(), v.ravel()], axis=-1).astype(float)
        if distort:
            uv = self.undistort_points(uv)
        s_map = self._pixels_to_cam_frame(uv, depth=1.0).reshape(self.res[0], self.res[1], 3)
        if mode == "normalised":
            s_map = s_map / np.linalg.norm(s_map, axis=-1, keepdims=True)
        self.sensor_map = s_map
        self.world_sensor_map = self._compute_world_sensor_map()

    def _pixels_to_cam_frame(self, uv: np.ndarray, depth: float | np.ndarray = 1.0) -> np.ndarray:
        """
        The camera frame point at ``depth`` behind each undistorted pixel.

        :param uv: undistorted pixel coordinates, (n, 2)
        :param depth: camera frame z to place the points at
        :return: an (n, 3) array of camera frame points
        """
        m_x, m_y = self.magnification
        c_x, c_y = self.principal_point
        depth = np.broadcast_to(np.asarray(depth, dtype=float), (len(uv),))
        # undo the telecentricity error at the depth being asked for
        scale = 1.0 + self.telecentricity * depth
        return np.stack([
            (uv[:, 0] - c_x) * scale / m_x,
            (uv[:, 1] - c_y) * scale / m_y,
            depth,
        ], axis=-1)

    def im_to_world_ray(self, cord, depth_im=None, distort=True, use_vector=False):
        """
        Given an image coordinate in opencv cords, nx2, returns a world point on its ray.

        The rays of a telecentric camera are parallel, so unlike the pinhole
        case the returned points do not share an origin -- each pixel's ray
        starts at its own place on the entrance plane.

        :param cord: points to project
        :param depth_im: a depth image, used to set the depth if given
        :param distort: whether the coordinate needs undistorting first
        :param use_vector: kept for signature compatibility; the telecentric
            map is computed directly either way
        :return: an (n, 3) array of world points
        """
        cord = np.asarray(cord)
        if cord.ndim == 1:
            cord = cord[None, ...]
        uv = cord.astype(float)
        if distort:
            uv = self.undistort_points(uv)

        depth = 1.0
        if depth_im is not None:
            depth = depth_im[cord[:, 1].astype(int), cord[:, 0].astype(int)]
            if np.any(np.isnan(depth)):
                logger.warning('Nan length found in depth image used for ray')

        cam_pts = self._pixels_to_cam_frame(uv, depth=depth)
        return h_tform(cam_pts, self.cam_to_world)

    # geometry for display

    def depth_for_magnification_error(self, tolerance: float = 1e-3) -> float:
        """
        The depth over which magnification stays within ``tolerance``.

        This is the one depth the calibration actually measures.  Magnification
        scales as ``1/(1 + eps*z)``, so a fractional error budget of
        ``tolerance`` is met out to ``|z| <= tolerance/|eps|``.  A perfectly
        telecentric lens has no such limit, which is why this is not the
        default drawing depth.

        :param tolerance: the fractional magnification error allowed
        :return: the usable full depth, or infinity for a perfect lens
        """
        if self.telecentricity == 0:
            return float('inf')
        return float(2.0 * tolerance / abs(self.telecentricity))

    @property
    def view_depth(self) -> float:
        """
        How deep to draw the imaged volume, in world units.

        Unlike a pinhole frustum, a telecentric camera's imaged cross section is
        fixed by the optics, so only the depth is a display choice.  It defaults
        to the depth over which magnification stays within
        ``MAGNIFICATION_ERROR_BUDGET`` -- the working volume the calibration
        actually measures, through ``eps``.

        A near-perfect lens has no such limit, and an unbounded box is no use in
        a rig plot, so the depth is capped at ``VIEW_DEPTH_CAP_FIELDS`` field
        widths.  A box drawn at the cap means "at least this deep", not "this
        deep".  Set the attribute to describe a real working volume instead.
        """
        if self._view_depth is not None:
            return self._view_depth
        cap = VIEW_DEPTH_CAP_FIELDS * float(np.mean(self.field_size))
        return min(self.depth_for_magnification_error(MAGNIFICATION_ERROR_BUDGET), cap)

    @view_depth.setter
    def view_depth(self, value: float | None) -> None:
        self._view_depth = None if value is None else float(value)

    def _view_box(self, length: float | None = None, centre: float = 0.0) -> np.ndarray:
        """
        The eight corners, in world coordinates, of the imaged volume.

        The box is centred on the camera frame origin rather than starting
        there, because a telecentric camera's position along its own axis is
        not observable -- so the origin is a label, not a lens.

        :param length: the depth to draw, defaulting to :attr:`view_depth`
        :param centre: the camera frame depth to centre the box on
        """
        length = self.view_depth if length is None else float(length)
        corners = np.array([
            [0, 0], [self.res[0], 0], [self.res[0], self.res[1]], [0, self.res[1]],
        ], dtype=float)
        return np.concatenate([
            h_tform(self._pixels_to_cam_frame(corners, depth=centre - length / 2),
                    self.cam_to_world),
            h_tform(self._pixels_to_cam_frame(corners, depth=centre + length / 2),
                    self.cam_to_world),
        ], axis=0)

    def get_mesh(self, scale=None):
        """
        Returns a box showing the imaged volume of the camera.

        A telecentric camera images a rectangular prism, not a frustum, so the
        cone the pinhole model draws would misrepresent it at every depth.  The
        cross section is set by the optics and does not scale; ``scale`` sets
        only the depth.

        :param scale: the depth to draw, defaulting to :attr:`view_depth`
        :return mesh: A PV mesh detailing the camera
        """
        _require_pyvista()
        return self._box_mesh(self._view_box(scale))

    def get_viewcone(self, scale=None, view_len=None, triangle=False):
        """
        Returns the imaged volume of the camera as a box.

        :param scale: unused; a telecentric cross section is fixed by the optics
        :param view_len: the depth to draw, defaulting to :attr:`view_depth`
        :param triangle: forces the mesh to only use triangular faces
        :returns mesh: A PV mesh showing the camera object's imaged volume
        """
        _require_pyvista()
        return self._box_mesh(self._view_box(view_len), triangle=triangle)

    @staticmethod
    def _box_mesh(verts: np.ndarray, triangle: bool = False):
        """Builds a PolyData box from eight corners, near face first."""
        quads = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
        ]
        if triangle:
            faces = np.hstack([[3, q[0], q[1], q[2], 3, q[0], q[2], q[3]] for q in quads])
        else:
            faces = np.hstack([[4, *q] for q in quads])
        return pv.PolyData(verts, faces)

    # resizing

    def scale_self_2n(self, down_scale_factor=1):
        """
        Downscales the camera, carrying the distortion coefficient with it.

        The division model acts on pixel offsets from the principal point, so
        halving the image halves those offsets and quarters the squared radius.
        The base class only rescales the intrinsic matrix, which for this model
        would leave the distortion describing an image that no longer exists.

        :param down_scale_factor: the power of two by which to scale
        """
        sf = float(-down_scale_factor)
        super().scale_self_2n(down_scale_factor)
        self.distortion_coefs = self.distortion_coefs * (2.0 ** (-2.0 * sf))
        self._update_state()

    def to_MVSnet_txt(self, *args, **kwargs):
        raise NotImplementedError(
            "The MVSNet camera format describes a pinhole camera, and a "
            "telecentric projection has no representation in it."
        )

    # parameter packing

    def to_param_vector(self) -> np.ndarray:
        """
        This camera's intrinsics in the order ``telecentric_intrinsic`` reads them.

        :return: ``[m_x, c_x, m_y, c_y, k, eps]``
        """
        return np.array([
            self.intrinsic[0, 0], self.intrinsic[0, 2],
            self.intrinsic[1, 1], self.intrinsic[1, 2],
            float(np.reshape(self.distortion_coefs, -1)[0]),
            self.telecentricity,
        ])

    def from_param_vector(self, params: np.ndarray) -> None:
        """
        Writes intrinsics back in the layout :meth:`to_param_vector` produced.

        :param params: ``[m_x, c_x, m_y, c_y, k, eps]``
        """
        intrinsic = np.eye(3)
        intrinsic[0, 0] = params[0]
        intrinsic[0, 2] = params[1]
        intrinsic[1, 1] = params[2]
        intrinsic[1, 2] = params[3]
        self.intrinsic = intrinsic
        self.distortion_coefs = np.array([params[4]])
        self.telecentricity = float(params[5])
