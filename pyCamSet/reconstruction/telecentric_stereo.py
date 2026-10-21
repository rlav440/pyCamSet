"""
Dense stereo for pairs of telecentric cameras, rectified for a learned matcher.

A telecentric camera is affine: its rays are parallel, so its epipolar lines
are parallel too, and rectification is a 2D affine warp of each image rather
than the homography a pinhole pair needs.  ``cv2.stereoRectify`` cannot be
used, because it builds that homography from camera centres that a
telecentric camera does not have.

The construction is exact for a perfect lens.  The two viewing directions
``d_0`` and ``d_1`` span every epipolar plane, so its normal
``n = d_0 x d_1`` is shared by both cameras and becomes the rectified row
axis: a point's row is ``s * n.X`` in both images.  Each camera's column axis
is ``x_k = n x d_k``, which lies in that camera's image plane, so it is
reachable by rotating and scaling the image about its own optical axis.
Disparity is then ``s * (x_L - x_R).X`` plus a constant -- affine in the
point, which is why a telecentric pair has no depth-dependent disparity
range to worry about beyond the depth of the scene itself.

A lens with residual telecentricity error ``eps`` is weakly perspective.
The warp is built from its affine part, which leaves a small row error that
is measured and reported as :attr:`TelecentricRectification.max_row_error`.
Triangulation always uses the full model, so the error costs match quality,
not geometric accuracy.

The matcher is any callable taking two rectified ``(h, w, 3)`` images and
returning a left-image disparity with ``u_right = u_left - disparity``, which
is the convention FoundationStereo uses, and OpenCV's block matchers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from pyCamSet.cameras.telecentric_camera import TelecentricCamera

logger = logging.getLogger(__name__)

#: disparity, in rectified pixels, given to the far end of the scene depth
DEFAULT_DISPARITY_MARGIN = 16.0
#: rectified row error, in pixels, past which the affine warp is flagged
ROW_ERROR_WARNING = 0.5
#: left-right disagreement, in rectified pixels, past which a match is dropped
DEFAULT_LR_TOLERANCE = 1.0
#: samples per image edge when finding the rectified image bounds
_EDGE_SAMPLES = 64

Matcher = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _image_border(res) -> np.ndarray:
    """Pixels sampled along the border of a ``res`` image, as (n, 2)."""
    w, h = res
    t = np.linspace(0.0, 1.0, _EDGE_SAMPLES)
    return np.concatenate([
        np.stack([t * (w - 1), np.zeros_like(t)], axis=-1),
        np.stack([t * (w - 1), np.full_like(t, h - 1)], axis=-1),
        np.stack([np.zeros_like(t), t * (h - 1)], axis=-1),
        np.stack([np.full_like(t, w - 1), t * (h - 1)], axis=-1),
    ])


def principal_ray(cam: TelecentricCamera) -> tuple[np.ndarray, np.ndarray]:
    """The world origin and unit direction of the ray through the principal point."""
    ends = cam.im_to_world_ray(np.array([cam.principal_point, cam.principal_point]),
                               distort=False)
    # im_to_world_ray places every point at depth one; move one to depth zero
    origin = ends[0] - cam.extrinsic[2, :3]
    return origin, _unit(cam.extrinsic[2, :3])


def closest_point_between_rays(o0, d0, o1, d1) -> np.ndarray:
    """The midpoint of the shortest segment joining two rays."""
    w = o0 - o1
    b = d0 @ d1
    denom = 1.0 - b * b
    if denom < 1e-12:
        return 0.5 * (o0 + o1)
    t0 = (b * (d1 @ w) - (d0 @ w)) / denom
    t1 = ((d1 @ w) - b * (d0 @ w)) / denom
    return 0.5 * (o0 + t0 * d0 + o1 + t1 * d1)


def triangulate_pair(cam0: TelecentricCamera, cam1: TelecentricCamera,
                     uv0: np.ndarray, uv1: np.ndarray, distort: bool = True) -> np.ndarray:
    """
    Triangulates matched pixels between two cameras, vectorised over points.

    Each camera contributes the two rows of the linear triangulation,
    ``(P_i - u_i P_2) . [X, 1] = 0``.  For a telecentric camera ``P_2`` is
    ``[0, 0, eps, 1]`` in the camera frame, so these rows are the pixel
    residuals themselves and the least squares solution is the reprojection
    optimum for a perfect lens, and very nearly so otherwise.

    :param cam0: the camera that saw ``uv0``
    :param cam1: the camera that saw ``uv1``
    :param uv0: (n, 2) pixel coordinates in ``cam0``
    :param uv1: (n, 2) pixel coordinates in ``cam1``
    :param distort: whether the coordinates still carry lens distortion
    :return: (n, 3) world points
    """
    if distort:
        uv0 = cam0.undistort_points(uv0)
        uv1 = cam1.undistort_points(uv1)
    rows = []
    for cam, uv in ((cam0, uv0), (cam1, uv1)):
        p = cam.proj
        rows.append(p[0][None] - uv[:, :1] * p[2][None])
        rows.append(p[1][None] - uv[:, 1:] * p[2][None])
    a = np.stack(rows, axis=1)  # (n, 4, 4)
    lhs, rhs = a[..., :3], -a[..., 3]
    ata = np.einsum('nki,nkj->nij', lhs, lhs)
    atb = np.einsum('nki,nk->ni', lhs, rhs)
    return np.linalg.solve(ata, atb[..., None])[..., 0]


def fixed_camera_params(cams) -> dict:
    """
    ``fixed_params`` holding every camera of a set at its calibration.

    The values are in the layout the camera's kernel blocks read, which is
    how a bundle handler writes a fixed camera straight into its primitive:
    the intrinsic block's own parameter vector, and the Rodrigues rotation
    then translation, cut to the length of the extrinsic block.
    """
    from pyCamSet.optimisation.camera_models import blocks_for_camset
    from pyCamSet.utils.general_utils import ext_4x4_to_rod

    _, extr_block = blocks_for_camset(cams)
    n_extr = extr_block.params.n_params
    return {cam.name: {"int": cam.to_param_vector(),
                       "ext": np.concatenate(ext_4x4_to_rod(cam.extrinsic))[:n_extr]}
            for cam in cams}


def bundle_adjust_points(cams, uvs, points: np.ndarray, threads: int = 1,
                         max_nfev: int = 20) -> np.ndarray:
    """
    Refines points by bundle adjustment, with the cameras held at their calibration.

    The problem is pyCamSet's free point bundle adjustment with every camera
    fixed, so only the points move, and the residual is the full lens model's
    reprojection error in the pixels the cameras recorded -- distortion and
    telecentricity error included -- rather than :func:`triangulate_pair`'s
    linearised, undistorted stand-in for it.

    :param cams: the cameras, one per entry of ``uvs``
    :param uvs: (n, 2) observed pixel coordinates in each camera; row ``i`` of
        each is the same point
    :param points: (n, 3) starting points, e.g. from :func:`triangulate_pair`
    :param threads: evaluation threads for the compiled kernels
    :param max_nfev: the solver's evaluation limit
    :return: the refined (n, 3) points
    """
    from pyCamSet.calibration_targets import ImageDetection, TargetDetection
    from pyCamSet.cameras import CameraSet
    from pyCamSet.optimisation.free_point_handler import (
        FreePointBundleHandler, FreePointTarget)
    from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment

    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points.reshape(0, 3)
    camset = CameraSet(camera_dict={cam.name: cam for cam in cams})
    keys = np.arange(len(points))
    detection = TargetDetection(cam_names=camset.get_names())
    for cam, uv in zip(cams, uvs):
        detection.add_detection(cam.name, 0, ImageDetection(keys=keys, image_points=uv))

    handler = FreePointBundleHandler(
        camset, FreePointTarget(points[None]), detection,
        fixed_params=fixed_camera_params(camset),
        options={"verbosity": 0, "outliers": "n", "max_nfev": max_nfev},
    )
    # every camera is fixed, so the parameter vector is the points alone
    handler.set_initial_params(points.reshape(-1).copy())
    optimisation, _ = run_bundle_adjustment(handler, threads=threads)
    return np.asarray(handler.get_updated_points(optimisation.x)).copy()


@dataclass
class TelecentricRectification:
    """
    The rectifying warp of a telecentric pair, and its way back to 3D.

    ``left`` and ``right`` are the input cameras in the order the matcher
    must see them, which is not necessarily the order they were given in;
    ``swapped`` records whether they were exchanged.  Each camera's warp takes
    an undistorted source pixel ``p`` to ``A @ p + b`` in the rectified image.
    """
    left: TelecentricCamera
    right: TelecentricCamera
    swapped: bool
    size: tuple[int, int]  # rectified (width, height)
    magnification: float  # rectified px per world unit
    affine_left: np.ndarray  # (2, 3)
    affine_right: np.ndarray  # (2, 3)
    row_axis: np.ndarray  # the shared epipolar plane normal, n
    centre: np.ndarray
    depth_axis: np.ndarray  # unit bisector, pointing away from the cameras
    max_row_error: float = 0.0
    _maps: dict = field(default_factory=dict, repr=False)

    # warps

    @staticmethod
    def _inverse(affine: np.ndarray, uv_rect: np.ndarray) -> np.ndarray:
        a, b = affine[:, :2], affine[:, 2]
        return (uv_rect - b) @ np.linalg.inv(a).T

    def rectified_to_source(self, uv_rect: np.ndarray, side: str,
                            distort: bool = True) -> np.ndarray:
        """
        Maps rectified pixels back to source pixels of one camera.

        :param uv_rect: (n, 2) rectified pixel coordinates
        :param side: 'left' or 'right'
        :param distort: return the distorted pixel the camera actually recorded
        :return: (n, 2) pixel coordinates in that camera's image
        """
        cam = self.left if side == 'left' else self.right
        affine = self.affine_left if side == 'left' else self.affine_right
        uv = self._inverse(affine, uv_rect)
        if distort:
            c = cam.principal_point
            xs, ys = cam._distort(uv[:, 0] - c[0], uv[:, 1] - c[1])
            uv = np.stack([xs + c[0], ys + c[1]], axis=-1)
        return uv

    def source_to_rectified(self, uv: np.ndarray, side: str,
                            distort: bool = True) -> np.ndarray:
        """Maps source pixels of one camera into the rectified image."""
        cam = self.left if side == 'left' else self.right
        affine = self.affine_left if side == 'left' else self.affine_right
        if distort:
            uv = cam.undistort_points(uv)
        return uv @ affine[:, :2].T + affine[:, 2]

    def _map(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        if side not in self._maps:
            w, h = self.size
            cols, rows = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
            src = self.rectified_to_source(
                np.stack([cols.ravel(), rows.ravel()], axis=-1), side)
            self._maps[side] = (src[:, 0].reshape(h, w).astype(np.float32),
                                src[:, 1].reshape(h, w).astype(np.float32))
        return self._maps[side]

    def rectify(self, im0: np.ndarray, im1: np.ndarray,
                interpolation=cv2.INTER_LINEAR) -> tuple[np.ndarray, np.ndarray]:
        """
        Warps an image pair, given in the order the cameras were given in.

        :return: the (left, right) rectified images, in matcher order
        """
        im_l, im_r = (im1, im0) if self.swapped else (im0, im1)
        out = []
        for side, im in (('left', im_l), ('right', im_r)):
            mx, my = self._map(side)
            out.append(cv2.remap(im, mx, my, interpolation,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0))
        return out[0], out[1]

    def valid_left(self) -> np.ndarray:
        """Rectified left pixels that draw from inside the left image."""
        mx, my = self._map('left')
        w, h = self.left.res
        return (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)

    # back to 3D

    def disparity_to_correspondences(self, disparity: np.ndarray,
                                     mask: np.ndarray | None = None,
                                     distort: bool = True):
        """
        Turns a left rectified disparity into matched source pixels.

        Pixels whose match lands outside either source image are dropped.

        :param disparity: (h, w) disparity, ``u_right = u_left - disparity``
        :param mask: optional (h, w) boolean mask of left pixels to keep
        :param distort: return distorted source pixels, as the camera recorded them
        :return: ``(uv_left, uv_right, keep)``, where ``keep`` is the (h, w)
            mask of rectified left pixels the two (n, 2) arrays came from
        """
        w, h = self.size
        if disparity.shape != (h, w):
            raise ValueError(
                f"disparity is {disparity.shape}, the rectified pair is {(h, w)}")
        keep = self.valid_left() & np.isfinite(disparity)
        if mask is not None:
            keep &= mask
        rows, cols = np.nonzero(keep)
        u_r = cols - disparity[rows, cols]
        uv_rect_l = np.stack([cols, rows], axis=-1).astype(float)
        uv_rect_r = np.stack([u_r, rows], axis=-1).astype(float)
        uv_l = self.rectified_to_source(uv_rect_l, 'left', distort=distort)
        uv_r = self.rectified_to_source(uv_rect_r, 'right', distort=distort)

        # rectified_to_source always returns the distorted pixel for the bounds
        # check, since that is the one that has to lie on the sensor
        uv_r_dist = uv_r if distort else self.rectified_to_source(uv_rect_r, 'right')
        rw, rh = self.right.res
        inside = ((u_r >= 0) & (uv_r_dist[:, 0] >= 0) & (uv_r_dist[:, 0] <= rw - 1)
                  & (uv_r_dist[:, 1] >= 0) & (uv_r_dist[:, 1] <= rh - 1))
        keep[rows[~inside], cols[~inside]] = False
        return uv_l[inside], uv_r[inside], keep

    def disparity_to_points(self, disparity: np.ndarray,
                            mask: np.ndarray | None = None,
                            bundle_adjust: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Triangulates a left rectified disparity with the full camera model.

        :param bundle_adjust: refine the linear solution by minimising the
            reprojection error in the recorded pixels, see
            :func:`bundle_adjust_points`
        :return: ``(points, keep)``: (n, 3) world points, and the (h, w) mask of
            rectified left pixels they came from, in row-major order
        """
        uv_l, uv_r, keep = self.disparity_to_correspondences(disparity, mask, distort=False)
        points = triangulate_pair(self.left, self.right, uv_l, uv_r, distort=False)
        if bundle_adjust:
            rec_l, rec_r, _ = self.disparity_to_correspondences(disparity, mask, distort=True)
            points = bundle_adjust_points((self.left, self.right), (rec_l, rec_r), points)
        return points, keep

    def expected_disparity(self, points: np.ndarray) -> np.ndarray:
        """The disparity a perfect lens pair would see for world points, (n,)."""
        points = np.atleast_2d(points)
        uv_l = self.source_to_rectified(self.left.project_points(points), 'left')
        uv_r = self.source_to_rectified(self.right.project_points(points), 'right')
        return uv_l[:, 0] - uv_r[:, 0]


def _rectified_axes(cam: TelecentricCamera, n: np.ndarray, s: float) -> np.ndarray:
    """
    The 2x2 map from an undistorted pixel offset to the rectified axes.

    A pixel offset ``(p - c)/m`` is the point's camera frame x, y.  The
    rectified axes ``x_k`` and ``n`` both lie in the image plane, so their
    camera frame components have no z, and the map is a rotation and scale.
    """
    r = cam.extrinsic[:3, :3]
    x_k = np.cross(n, r[2])
    m = cam.magnification
    return s * np.stack([(r @ x_k)[:2] / m, (r @ n)[:2] / m])


def rectify_telecentric_pair(cam0: TelecentricCamera, cam1: TelecentricCamera,
                             centre: np.ndarray | None = None,
                             depth_range: float | None = None,
                             scale: float = 1.0,
                             disparity_margin: float = DEFAULT_DISPARITY_MARGIN,
                             ) -> TelecentricRectification:
    """
    Builds the affine rectification of a telecentric pair.

    Which camera ends up on the left is decided by the geometry, not the
    argument order: the matcher is trained on pairs where disparity grows
    toward the cameras, so the pair is ordered to make that true, and the row
    axis is chosen to keep both images as close to upright as it can.

    :param cam0: one telecentric camera
    :param cam1: the other
    :param centre: a world point in the middle of the scene; defaults to the
        closest approach of the two principal rays
    :param depth_range: the scene depth, in world units, along the bisector of
        the two views; defaults to the mean field width
    :param scale: rectified magnification relative to the mean of the two
        cameras', below one to shrink the matcher's input
    :param disparity_margin: rectified disparity given to the far end of the
        depth range, keeping every disparity positive
    :return: the rectification
    """
    for cam in (cam0, cam1):
        if not isinstance(cam, TelecentricCamera):
            raise TypeError(f"{cam.name} is a {type(cam).__name__}, not a TelecentricCamera")

    d0, d1 = cam0.extrinsic[2, :3], cam1.extrinsic[2, :3]
    normal = np.cross(d0, d1)
    sin_angle = np.linalg.norm(normal)
    if sin_angle < 1e-3:
        raise ValueError(
            f"{cam0.name} and {cam1.name} view along nearly the same direction "
            f"({np.degrees(np.arcsin(sin_angle)):.3f} deg apart); parallel "
            "telecentric views have no disparity to measure")
    n = normal / sin_angle
    # the rectified row axis runs down the image, so keep it near both cameras' own
    if n @ (cam0.extrinsic[1, :3] + cam1.extrinsic[1, :3]) < 0:
        n = -n

    # disparity grows toward the cameras when n . (d_L x d_R) < 0
    swapped = bool(n @ np.cross(d0, d1) > 0)
    left, right = (cam1, cam0) if swapped else (cam0, cam1)
    d_l, d_r = left.extrinsic[2, :3], right.extrinsic[2, :3]

    if centre is None:
        centre = closest_point_between_rays(*principal_ray(left), *principal_ray(right))
    centre = np.asarray(centre, dtype=float)
    if depth_range is None:
        depth_range = float(np.mean([left.field_size.mean(), right.field_size.mean()]))
    depth_axis = _unit(d_l + d_r)

    s = scale * float(np.mean([left.magnification.mean(), right.magnification.mean()]))
    lin_l = _rectified_axes(left, n, s)
    lin_r = _rectified_axes(right, n, s)

    # choose the offsets so both cameras put the centre on the same row, with
    # the left image's content starting at column zero and the pair's rows
    # cropped to the band both cameras see
    def rect_of(cam, lin, uv_undistorted):
        return (uv_undistorted - cam.principal_point) @ lin.T

    centre_l = rect_of(left, lin_l, left.project_points(centre, distort=False))[0]
    centre_r = rect_of(right, lin_r, right.project_points(centre, distort=False))[0]
    border_l = rect_of(left, lin_l, left.undistort_points(_image_border(left.res))) - centre_l
    border_r = rect_of(right, lin_r, right.undistort_points(_image_border(right.res))) - centre_r

    v_min = max(border_l[:, 1].min(), border_r[:, 1].min())
    v_max = min(border_l[:, 1].max(), border_r[:, 1].max())
    if v_max <= v_min:
        raise ValueError(f"{left.name} and {right.name} share no epipolar lines")

    # with both centred, disparity(X) = s * (x_L - x_R) . (X - centre); the
    # far end of the depth range is set to the margin
    x_l, x_r = np.cross(n, d_l), np.cross(n, d_r)
    far = centre + 0.5 * depth_range * depth_axis
    d_far = s * (x_l - x_r) @ (far - centre)
    u0_l = -border_l[:, 0].min()
    u0_r = u0_l - (disparity_margin - d_far)
    # a shared shift leaves disparity alone; use it so neither image starts
    # at a negative column and loses the part of the view that falls there
    shift = max(0.0, -(border_r[:, 0].min() + u0_r))
    u0_l, u0_r = u0_l + shift, u0_r + shift

    offset_l = np.array([u0_l, -v_min]) - centre_l
    offset_r = np.array([u0_r, -v_min]) - centre_r
    affine_l = np.hstack([lin_l, (offset_l - lin_l @ left.principal_point)[:, None]])
    affine_r = np.hstack([lin_r, (offset_r - lin_r @ right.principal_point)[:, None]])

    width = int(np.ceil(max(border_l[:, 0].max() + u0_l, border_r[:, 0].max() + u0_r)))
    height = int(np.ceil(v_max - v_min))

    rect = TelecentricRectification(
        left=left, right=right, swapped=swapped, size=(width, height),
        magnification=s, affine_left=affine_l, affine_right=affine_r,
        row_axis=n, centre=centre, depth_axis=depth_axis,
    )
    rect.max_row_error = _row_error(rect, depth_range)
    if rect.max_row_error > ROW_ERROR_WARNING:
        logger.warning(
            "%s/%s: the affine rectification leaves rows up to %.2f px apart "
            "over the depth range, from the lenses' telecentricity error; "
            "the matcher will see slightly misaligned epipolar lines",
            left.name, right.name, rect.max_row_error)
    return rect


def _row_error(rect: TelecentricRectification, depth_range: float) -> float:
    """The largest row mismatch the full model shows across the scene volume."""
    if rect.left.telecentricity == 0 and rect.right.telecentricity == 0:
        return 0.0
    n, z = rect.row_axis, rect.depth_axis
    x = _unit(np.cross(n, z))
    half = 0.5 * float(np.mean(rect.left.field_size))
    t = np.linspace(-1.0, 1.0, 7)
    a, b, c = np.meshgrid(t * half, t * half, t * 0.5 * depth_range, indexing='ij')
    pts = rect.centre + a.reshape(-1, 1) * x + b.reshape(-1, 1) * n + c.reshape(-1, 1) * z
    v_l = rect.source_to_rectified(rect.left.project_points(pts), 'left')[:, 1]
    v_r = rect.source_to_rectified(rect.right.project_points(pts), 'right')[:, 1]
    return float(np.max(np.abs(v_l - v_r)))


# the matcher

def as_matcher_image(im: np.ndarray) -> np.ndarray:
    """
    Converts an image to the (h, w, 3) float 0-255 layout the matcher expects.

    Grey images are repeated across channels, and 16 bit images are scaled
    by their own range rather than their type's, since scientific cameras
    rarely fill it.
    """
    im = np.asarray(im)
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, axis=-1)
    elif im.shape[-1] == 1:
        im = np.repeat(im, 3, axis=-1)
    elif im.shape[-1] == 4:
        im = im[..., :3]
    im = im.astype(np.float32)
    if im.max() > 255:
        im = 255.0 * (im - im.min()) / max(float(im.max() - im.min()), 1.0)
    return im


@dataclass
class TelecentricStereoResult:
    """A dense reconstruction from one telecentric pair."""
    points: np.ndarray  # (n, 3) world points
    colours: np.ndarray  # (n, c) left rectified image values at each point
    keep: np.ndarray  # (h, w) rectified left pixels that produced a point
    disparity: np.ndarray  # (h, w) the matcher's output
    rectified: tuple[np.ndarray, np.ndarray]
    rectification: TelecentricRectification
    consistent: np.ndarray | None = None  # (h, w) pixels passing the left-right check

    def to_pyvista(self):
        """The points as a PolyData, coloured by the left image."""
        import pyvista as pv
        cloud = pv.PolyData(self.points)
        cols = self.colours
        if cols.ndim == 2 and cols.shape[1] == 3:
            cloud['RGB'] = cols.astype(np.uint8)
        else:
            cloud['intensity'] = cols.reshape(len(self.points), -1)[:, 0]
        return cloud


def left_right_consistent(matcher: Matcher, rect_l: np.ndarray, rect_r: np.ndarray,
                          disparity: np.ndarray,
                          tolerance: float = DEFAULT_LR_TOLERANCE) -> np.ndarray:
    """
    Left pixels whose match, matched back from the right image, returns to them.

    A matcher gives every left pixel a disparity, including the ones on
    surface the right camera never saw, where the answer is extrapolated from
    the surroundings. Matching the other way round and checking the two agree
    finds those. The right image's disparity is the matcher's own output for
    the mirrored, swapped pair: mirroring turns the right view into a left one
    with the same, positive, disparity.

    :param matcher: ``matcher(left, right) -> disparity``
    :param rect_l: the left rectified image
    :param rect_r: the right rectified image
    :param disparity: the matcher's left disparity for the pair
    :param tolerance: the largest disagreement kept, in rectified pixels
    :return: (h, w) boolean mask of consistent left pixels
    """
    disp_r = np.asarray(matcher(np.ascontiguousarray(rect_r[:, ::-1]),
                                np.ascontiguousarray(rect_l[:, ::-1])), dtype=float)[:, ::-1]
    h, w = disparity.shape
    cols, rows = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    src = cols - disparity.astype(np.float32)
    # replicate rather than pad with NaN: an exact sample on the last row or
    # column still reads its zero-weight neighbour, and NaN * 0 is NaN
    back = cv2.remap(disp_r.astype(np.float32), src, rows, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    with np.errstate(invalid='ignore'):
        return (src >= 0) & (src <= w - 1) & (np.abs(disparity - back) < tolerance)


def reconstruct_telecentric_pair(cam0: TelecentricCamera, cam1: TelecentricCamera,
                                 im0: np.ndarray, im1: np.ndarray,
                                 matcher: Matcher,
                                 rectification: TelecentricRectification | None = None,
                                 mask: np.ndarray | None = None,
                                 bundle_adjust: bool = False,
                                 lr_tolerance: float | None = DEFAULT_LR_TOLERANCE,
                                 **rectify_kwargs) -> TelecentricStereoResult:
    """
    Rectifies a telecentric pair, matches it, and triangulates the result.

    :param cam0: the camera that took ``im0``
    :param cam1: the camera that took ``im1``
    :param im0: an image from ``cam0``
    :param im1: an image from ``cam1``
    :param matcher: ``matcher(left, right) -> disparity``
    :param rectification: a precomputed rectification of this pair, reused
        across frames to skip rebuilding the maps
    :param mask: optional (h, w) mask of rectified left pixels to reconstruct
    :param bundle_adjust: refine each point by minimising its reprojection
        error, rather than stopping at the linear triangulation
    :param lr_tolerance: drop pixels whose left and right disparities disagree
        by more than this, see :func:`left_right_consistent`; ``None`` skips
        the check and its second matcher pass
    :param rectify_kwargs: passed to :func:`rectify_telecentric_pair`
    :return: the points, and everything that went into them
    """
    rect = rectification or rectify_telecentric_pair(cam0, cam1, **rectify_kwargs)
    rect_l, rect_r = rect.rectify(im0, im1)
    disparity = np.asarray(matcher(rect_l, rect_r), dtype=float)
    consistent = None
    if lr_tolerance is not None:
        consistent = left_right_consistent(matcher, rect_l, rect_r, disparity, lr_tolerance)
        mask = consistent if mask is None else mask & consistent
    points, keep = rect.disparity_to_points(disparity, mask=mask, bundle_adjust=bundle_adjust)
    colours = rect_l[keep]
    return TelecentricStereoResult(
        points=points, colours=colours, keep=keep, disparity=disparity,
        rectified=(rect_l, rect_r), rectification=rect, consistent=consistent)
