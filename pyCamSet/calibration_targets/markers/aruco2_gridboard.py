"""
aruco2's ``GridBoard`` detector -- the ChArUco2 design.

Every square of the board carries an ArUco marker (a standard marker on a
black square, an inverted one on a white square), giving an N x M board
(N x M markers) and (N+1) x (M+1) observable intersection corners including
the board's own border. The design is described in
https://www.sciencedirect.com/science/article/pii/S2352711026003249.

``aruco2.detect_grid_board`` hands back a populated ``GridBoard`` object.
Its Python API exposes the found corners only indirectly, through
``aruco2.get_solve_pnp_points(board, marker_size)``, which returns matched
``(object_points, image_points)`` arrays for the corners it actually found.
Mapping a detection back to pyCamSet's stable per-corner index (matching
:data:`ChArUco2.point_data`'s row order) therefore has to go through the
object points, not through anything positional in ``markers``. Every
returned object point lies on the ``(grid_width+1) x (grid_height+1)``
intersection lattice at ``(col, row) * marker_size``, so::

    row = round(object_point[1] / marker_size)
    col = round(object_point[0] / marker_size)
    gid = row * (grid_width + 1) + col

recovers the same row-major index :func:`pyCamSet.calibration_targets
.charuco2.target.ChArUco2`'s own ``point_data`` is built with. This mapping
was verified empirically against a full board, a board with one quadrant
occluded, a 90-degree-rotated board, and a perspective-warped board: in
every case the surviving corners' recovered ``gid``s pointed at the same
physical 3-D point the full, undistorted board assigned them, with
sub-pixel image-point reprojection error.
"""

from __future__ import annotations

import logging
import warnings

import cv2
import numpy as np

from pyCamSet.calibration_targets.markers.aruco2 import (
    ARUCO2_AVAILABLE,
    _require_aruco2,
    _as_uint8_image,
)
from pyCamSet.calibration_targets.markers.backend_registry import (
    ARUCO2_BACKEND,
    marker_backend_available,
)
from pyCamSet.calibration_targets.core.parameters import (
    Parameter,
    DetectorParameterisation,
)

_LOG = logging.getLogger(__name__)

# Reuse the same lazy-import guard as the aruco2 marker backend
# (pyCamSet.calibration_targets.markers.aruco2, imported above): this module
# still needs its own local ``aruco2`` reference to call
# detect_grid_board/get_solve_pnp_points/etc, but must import cleanly even
# when the package is not installed.
try:
    import aruco2  # noqa: F401
except (ImportError, OSError):  # pragma: no cover - exercised by the mocked gate check
    aruco2 = None  # type: ignore[assignment]


def _grid_board_ids(grid_size: tuple[int, int], ids,
                    dict_int: int | None = None) -> list[int] | None:
    """
    A board's marker ids as aruco2 takes them, or None for its default ids.

    :param dict_int: the dictionary the ids index, to check them against;
        not checked when None.
    :raises ValueError: when ``ids`` does not give exactly one distinct
        whole-number id per square, or names a marker the dictionary does
        not have.
    """
    if ids is None:
        return None
    grid_w, grid_h = int(grid_size[0]), int(grid_size[1])
    raw = np.asarray(ids).reshape(-1)
    whole = raw.dtype.kind in "iu" or (
        raw.dtype.kind == "f" and bool(np.all(np.isfinite(raw)))
        and np.array_equal(raw, np.round(raw)))
    if not whole:
        # int() would truncate 0.9 to marker 0, printing and looking for a
        # marker nobody asked for.
        raise ValueError(f"Marker ids must be whole numbers; got {raw.tolist()[:5]}.")
    flat = [int(i) for i in raw]
    if len(flat) != grid_w * grid_h:
        raise ValueError(
            f"A {grid_w}x{grid_h} grid board has {grid_w * grid_h} marker "
            f"squares, so it needs that many ids; got {len(flat)}.")
    if len(set(flat)) != len(flat):
        repeated = sorted({i for i in flat if flat.count(i) > 1})
        raise ValueError(
            f"Every square of a grid board carries its own marker; ids "
            f"{repeated[:5]} are repeated.")
    if dict_int is not None:
        n_markers = dictionary_marker_count(dict_int)
        outside = [i for i in flat if not 0 <= i < n_markers]
        if outside:
            raise ValueError(
                f"Marker id(s) {outside[:5]} are outside this dictionary's "
                f"{n_markers} markers.")
    return flat


#: The grey level of the border an image is padded with before detection.
#: White, because aruco2's black-marker pass looks for pixels darker than
#: their neighbourhood's mean, and a pixel at 255 never is: the border adds
#: no black contour, so every black marker aruco2 finds comes from a contour
#: inside the original image. (No constant keeps the white pass out of the
#: border too -- a flat region always passes its "not darker" test -- but that
#: pass does not push corners outwards; see _grid_board_padding.)
_PAD_VALUE = 255

#: The largest half window aruco2 gives cv2.cornerSubPix:
#: ``min(int(3 * max(1, avrgLen / 34)), 9)`` at aruco2.cpp:742 and :1477.
_ARUCO2_MAX_SUBPIX_HALF_WINDOW = 9


def _aruco2_erosion_passes(width: int) -> int:
    """aruco2's ``maxErodeIterations`` for an image this many pixels wide
    (aruco2.cpp:997)."""
    return max(2, int(2.0 * width / 2000.0 + 0.5))


def _grid_board_padding(width: int) -> int:
    """
    How wide a white border keeps ``aruco2.detect_grid_board`` from raising
    on a board cut by the image edge.

    OpenCV's ``cornerSubPix`` asserts that every starting point lies inside
    the image (cornersubpix.cpp:99), and never moves a point out of it. The
    grid-board detector breaks that at its board-corner refinement
    (aruco2.cpp:1484): after refining a black marker's corners
    (aruco2.cpp:746), it pushes each one ``2 + i`` pixels further out along
    the marker's diagonal, for erosion pass ``i < maxErodeIterations``
    (aruco2.cpp:1002-1020) -- so a black marker touching the edge hands the
    next refinement a corner up to ``maxErodeIterations + 1`` pixels outside.

    With a white border ``p`` pixels wide every black contour lies in the
    original image, so at least ``p`` from the padded edge (see
    :data:`_PAD_VALUE`); the first refinement moves a corner at most its half
    window, 9 pixels, and the push adds at most ``maxErodeIterations + 1``.
    White markers are not pushed, and a board corner is an average of marker
    corners, so none lies further out. ``p >= maxErodeIterations + 10``
    therefore keeps every starting point inside; one pixel more absorbs
    float32 rounding. ``maxErodeIterations`` grows with the *padded* width,
    hence the loop.
    """
    pad = 0
    while True:
        needed = (_ARUCO2_MAX_SUBPIX_HALF_WINDOW
                  + _aruco2_erosion_passes(width + 2 * pad) + 2)
        if needed <= pad:
            return pad
        pad = needed


def _edge_margin() -> int:
    """
    How far inside the image a returned corner must lie, in pixels.

    ``cornerSubPix`` samples ``half_window + 1`` pixels either side of a
    corner (its window plus the one-pixel gradient border); aruco2's half
    window is at most :data:`_ARUCO2_MAX_SUBPIX_HALF_WINDOW`.
    """
    return _ARUCO2_MAX_SUBPIX_HALF_WINDOW + 1


def _is_cornersubpix_edge_failure(err: Exception) -> bool:
    """
    Whether ``err`` is OpenCV's ``cornerSubPix`` "starting point outside the
    image" assertion -- the one :func:`_grid_board_padding` exists to avoid,
    and the only failure :func:`detect_grid_board_corners` treats as "this
    board is lost", not "something is wrong".

    aruco2 hands the assertion back as a Python ``ValueError`` (verified
    against OpenCV 4.11.0, by calling ``aruco2.detect_grid_board`` unpadded
    on a board cut by the image edge)::

        OpenCV(4.11.0) .../cornersubpix.cpp:99: error: (-215:Assertion
        failed) Rect(0, 0, src.cols, src.rows).contains(cT) in function
        'cv::cornerSubPix'

    but a future OpenCV build could raise it as ``cv2.error`` instead, so
    both are caught upstream and only the message is checked here. Any other
    ``ValueError``/``cv2.error`` -- a malformed board, an unreadable image,
    or even a *different* ``cornerSubPix`` assertion (e.g. a bad window
    size) -- is not this failure and must propagate: both substrings are
    required, not either alone.
    """
    text = str(err).lower()
    return "cornersubpix" in text and "contains(ct)" in text


#: Newton refinement's default smoothing (see :func:`refine_grid_board_corners`).
_REFINE_SIGMA = 1.8
#: The coarse-to-fine retry's starting smoothing, for corners the fine scale refuses.
_REFINE_COARSE_SIGMA = 3.0
#: Point-symmetry NCC a refined corner must reach to be accepted (an occluder
#: or another non-corner feature nearby fails this).
_REFINE_MIN_SYMMETRY = 0.3


def _lattice_coords(corner_ids: np.ndarray, grid_w: int) -> np.ndarray:
    """``corner_ids`` (``row * (grid_w+1) + col``) as ``(N, 2)`` ``(col, row)``."""
    ids = np.asarray(corner_ids, dtype=np.int64)
    width = int(grid_w) + 1
    return np.stack([ids % width, ids // width], axis=-1)


def _local_square_px(lattice_cr: np.ndarray, image_points: np.ndarray) -> np.ndarray:
    """
    The lattice spacing at each corner, in pixels: the median distance to its
    detected 4-neighbours (up, down, left, right on the lattice), or the
    board's own median where a corner has none (a single detected corner, or
    one on its own after occlusion). A neighbour -- or the corner itself --
    with a non-finite position is skipped, so one NaN corner cannot poison
    its neighbours' spacing estimate.
    """
    index = {(int(c), int(r)): i for i, (c, r) in enumerate(lattice_cr)}
    finite = np.isfinite(image_points).all(axis=1)
    size = np.full(len(lattice_cr), np.nan)
    for i, (c, r) in enumerate(lattice_cr):
        if not finite[i]:
            continue
        dists = [
            float(np.hypot(*(image_points[i] - image_points[index[key]])))
            for key in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1))
            if key in index and finite[index[key]]
        ]
        if dists:
            size[i] = np.median(dists)
    fallback = np.nanmedian(size) if np.isfinite(size).any() else 20.0
    return np.where(np.isfinite(size), size, fallback)


def _bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinearly sample a float32 image at (possibly fractional) points,
    clamped to stay inside it."""
    h, w = image.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    ix = x.astype(np.int64)
    iy = y.astype(np.int64)
    fx = x - ix
    fy = y - iy
    return (image[iy, ix] * (1 - fx) * (1 - fy) + image[iy, ix + 1] * fx * (1 - fy)
            + image[iy + 1, ix] * (1 - fx) * fy + image[iy + 1, ix + 1] * fx * fy)


def _saddle_derivatives(roi: np.ndarray, sigma: float) -> tuple[np.ndarray, ...]:
    """First and second derivatives of ``roi`` smoothed with a Gaussian of
    this ``sigma``, as ``(Ix, Iy, Ixx, Ixy, Iyy)``."""
    smoothed = cv2.GaussianBlur(
        roi.astype(np.float32), (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    return (
        cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8),
        cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8),
        cv2.Sobel(smoothed, cv2.CV_32F, 2, 0, ksize=3, scale=1 / 4),
        cv2.Sobel(smoothed, cv2.CV_32F, 1, 1, ksize=3, scale=1 / 4),
        cv2.Sobel(smoothed, cv2.CV_32F, 0, 2, ksize=3, scale=1 / 4),
    )


def _newton_to_saddle(
    derivatives: tuple[np.ndarray, ...], points: np.ndarray, max_move: np.ndarray,
    iterations: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Newton-iterate each point to the nearest saddle point of the smoothed
    image its derivatives come from.

    A saddle of the image's light is a saddle of any monotone tone curve
    applied to it -- unlike ``cv2.cornerSubPix``, which chases the image's
    own second moments and so drifts with gamma -- which is why this, not a
    bigger ``cornerSubPix`` window, is what recovers accuracy under blur
    (validated in stage A task A3: RMS 1.40 -> 0.21 px on a blurred,
    gamma-2.2 ChArUco2 render).

    :param derivatives: as :func:`_saddle_derivatives` returns.
    :param points: ``(N, 2)`` starting positions, in the derivative images'
        own coordinates.
    :param max_move: ``(N,)`` cap on total movement per point, in pixels.
    :return: ``(refined points, accepted mask)``. A point is refused --
        returned unmoved, ``False`` in the mask -- when the Hessian never
        becomes a saddle (``det(H) >= 0``, so a local minimum, maximum or
        ridge, not a corner) or it would move further than ``max_move``.
    """
    ix, iy, ixx, ixy, iyy = derivatives
    start = points.copy()
    p = points.copy()
    ok = np.ones(len(p), dtype=bool)
    for _ in range(iterations):
        gx = _bilinear_sample(ix, p[:, 0], p[:, 1])
        gy = _bilinear_sample(iy, p[:, 0], p[:, 1])
        hxx = _bilinear_sample(ixx, p[:, 0], p[:, 1])
        hxy = _bilinear_sample(ixy, p[:, 0], p[:, 1])
        hyy = _bilinear_sample(iyy, p[:, 0], p[:, 1])
        det = hxx * hyy - hxy * hxy
        ok &= det < 0.0  # a saddle, not a blob or a smooth slope
        safe_det = np.where(det < 0.0, det, -1.0)
        step = np.stack(
            [-(hyy * gx - hxy * gy) / safe_det, -(hxx * gy - hxy * gx) / safe_det], axis=1)
        length = np.hypot(step[:, 0], step[:, 1])
        step *= np.minimum(1.0, 1.0 / np.maximum(length, 1e-12))[:, None]  # at most 1 px/iteration
        p = np.where(ok[:, None], p + step, p)
        if np.all(length < 1e-3):
            break
    ok &= np.hypot(*(p - start).T) <= max_move
    return np.where(ok[:, None], p, start), ok


def _point_symmetry(grey: np.ndarray, points: np.ndarray, square_px: np.ndarray) -> np.ndarray:
    """
    Normalised cross-correlation between the patch at each point and that
    patch rotated 180 degrees.

    A genuine saddle corner (an X of alternating light and dark) is close to
    point-symmetric; a saddle found near an occluder or another non-corner
    edge is not. This is a gate on refinement's own result, not a validator
    of aruco2's corners in general (a wrong corner can still look symmetric).
    """
    out = np.zeros(len(points))
    h, w = grey.shape
    for i, (point, square) in enumerate(zip(points, square_px)):
        radius = int(np.clip(round(0.15 * square), 3, 10))
        centre = (float(point[0]), float(point[1]))
        if not (radius <= centre[0] <= w - 1 - radius and radius <= centre[1] <= h - 1 - radius):
            continue  # too close to the ROI edge for a full patch: leave at 0 (refused)
        patch = cv2.getRectSubPix(grey, (2 * radius + 1, 2 * radius + 1), centre).astype(np.float64)
        patch -= patch.mean()
        rotated = patch[::-1, ::-1]
        denom = np.sqrt((patch * patch).sum() * (rotated * rotated).sum())
        out[i] = (patch * rotated).sum() / denom if denom > 1e-9 else 0.0
    return out


def refine_grid_board_corners(
    image: np.ndarray, lattice_cr: np.ndarray, image_points: np.ndarray,
    sigma: float = _REFINE_SIGMA, coarse_sigma: float = _REFINE_COARSE_SIGMA,
    min_symmetry: float = _REFINE_MIN_SYMMETRY,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Move each corner to the saddle point of the Gaussian-smoothed image
    nearest aruco2's own estimate.

    Works on the original (unpadded) grey image, over the detection's
    bounding box plus enough margin for the smoothing and Newton search --
    not the whole image, so cost scales with the board, not the frame.
    Corners the fine scale (``sigma``) refuses get one coarse-to-fine retry,
    starting at ``coarse_sigma`` and finishing at ``sigma``: this recovers
    most of the corners a single fine pass misses under blur, without the
    coarse scale's worse localisation on a sharp image (stage A task A3).
    The four outer corners of the board are not saddle points of a
    chequerboard signal, so both passes typically refuse them; they keep
    aruco2's own position, which is what this function's contract promises,
    not a defect. A refined corner is accepted only when it is also
    point-symmetric (see :func:`_point_symmetry`) -- otherwise, most often
    near an occluder, it keeps aruco2's position too.

    :param image: the uint8 (or greyscale-convertible) image to refine in.
    :param lattice_cr: ``(N, 2)`` ``(col, row)`` lattice coordinates of each
        point, e.g. from :func:`_lattice_coords`.
    :param image_points: ``(N, 2)`` aruco2's own corner positions.
    :return: ``(refined points, accepted mask)`` -- ``accepted`` is True
        where the point moved to a saddle within its movement cap and passed
        the symmetry gate; refused points are returned unmoved. A
        non-finite input point is always refused (and returned as-is): it
        cannot be searched from, and is excluded from the bounding box and
        from other corners' local-spacing estimate rather than raising.
    """
    grey = np.asarray(image)
    if grey.ndim == 3:
        grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
    pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts, np.zeros(0, dtype=bool)
    lattice_cr = np.asarray(lattice_cr, dtype=np.int64).reshape(-1, 2)
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        return pts, np.zeros(len(pts), dtype=bool)
    square = _local_square_px(lattice_cr, pts)

    # Derivatives only over the board's bounding box, padded enough for the
    # coarse Gaussian's support plus a working margin. The box is sized from
    # the finite points only, so one NaN corner cannot blow it up or shrink
    # it to nothing.
    pad = int(np.ceil(4 * coarse_sigma)) + 8
    x0 = max(int(pts[finite, 0].min()) - pad, 0)
    y0 = max(int(pts[finite, 1].min()) - pad, 0)
    x1 = min(int(pts[finite, 0].max()) + pad + 1, grey.shape[1])
    y1 = min(int(pts[finite, 1].max()) + pad + 1, grey.shape[0])
    roi = grey[y0:y1, x0:x1]
    # A non-finite point gets a dummy, in-bounds starting position purely so
    # the batched Newton search below has something to index into without
    # crashing; its result is discarded below regardless (refused, unmoved).
    centre = np.array([x0 + (x1 - x0) / 2.0, y0 + (y1 - y0) / 2.0])
    safe_pts = np.where(finite[:, None], pts, centre)
    local = safe_pts - (x0, y0)

    fine_move = np.clip(0.2 * square, 1.0, 3.0)
    fine, ok_fine = _newton_to_saddle(_saddle_derivatives(roi, sigma), local, fine_move)

    retry = ~ok_fine
    if np.any(retry):
        coarse_move = np.clip(0.25 * square, 1.0, 4.0)
        stage1, ok1 = _newton_to_saddle(
            _saddle_derivatives(roi, coarse_sigma), local[retry], coarse_move[retry])
        stage2, ok2 = _newton_to_saddle(_saddle_derivatives(roi, sigma), stage1, coarse_move[retry])
        moved = np.hypot(*(stage2 - local[retry]).T) <= coarse_move[retry]
        ok_retry = ok1 & ok2 & moved
        fine[retry] = np.where(ok_retry[:, None], stage2, local[retry])
        ok_fine[retry] = ok_retry

    refined = fine + (x0, y0)
    symmetry = _point_symmetry(grey.astype(np.float32), refined, square)
    accepted = ok_fine & (symmetry >= min_symmetry) & finite
    return np.where(accepted[:, None], refined, pts), accepted


#: How far (in lattice steps) a corner's leave-one-out neighbourhood reaches.
_VALIDATE_RADIUS = 2
#: Reject a corner when its local-homography residual exceeds
#: ``max(tau_abs, tau_rel * local_square_px) + model_uncertainty`` (see
#: :func:`validate_grid_board_corners`).
_VALIDATE_TAU_REL = 0.10
_VALIDATE_TAU_ABS = 2.0
_VALIDATE_IRLS_ITERATIONS = 3
#: A corner with fewer non-collinear neighbours than this cannot be checked,
#: and is dropped rather than trusted.
_VALIDATE_MIN_NEIGHBOURS = 5
#: How much the near/far local-homography disagreement (see
#: :func:`validate_grid_board_corners`) is scaled up before being added to
#: the removal tolerance. The near fit still extrapolates (just less), so
#: the raw near/far gap undershoots the true curvature-driven bias at the
#: far fit's own corner; calibrated empirically (see
#: tests/test_aruco2_gridboard_quality.py) against two competing numbers:
#: false removals on exact distorted lattices (which want it large) and
#: injected-fault detection (which wants it small, since the same extra
#: tolerance that forgives curvature can also forgive a small fault). 4.0
#: is the smallest value that clears every combination of the false-removal
#: grid at <=0.5%; injected-fault detection stays within stage A's
#: tolerance at this value but is not perfectly unchanged (a 10 px shift
#: lands within a few degrees of an edge, where curvature and fault
#: magnitude are closest, roughly 1-2% of the time).
_VALIDATE_MODEL_ERROR_SCALE = 4.0
#: The Tukey IRLS reweighting inside ``score`` exists to reject a genuinely
#: *wrong* neighbour (an injected fault elsewhere on the board) without
#: letting it drag the fit -- not to pass judgement on the corner under
#: test, which the final ``tau_abs``/``tau_rel``-based comparison already
#: does. Reusing that same tight, fixed threshold inside IRLS made it
#: misread ordinary curvature (tens of pixels of genuine, correlated
#: deviation from a single local homography, at a large board's strongly
#: distorted, steeply tilted edge) as if every neighbour were an outlier,
#: collapsing the fit onto a near-degenerate handful of survivors --
#: worse-conditioned than the honest full neighbourhood, and too few left
#: for the model-uncertainty check above to run at all. IRLS instead scales
#: its own threshold to a robust (low-percentile, not mean) residual among
#: the corner's own present neighbours each round -- a standard
#: robust-regression device (c.f. a MAD-scaled Tukey biweight): correlated
#: curvature moves most residuals together, so this tracks it and the
#: threshold loosens with it; a minority of genuinely wrong neighbours (an
#: injected fault) leaves it near zero, so the threshold -- and IRLS's power
#: to reject them -- does not loosen at all. A low percentile, not the
#: median, because a very small board's tiny neighbourhoods can have a
#: sizeable *fraction* corrupted (e.g. 3 of 16 corners on a 3x3 ghost-marker
#: board), which pulls the median itself up; a low percentile stays robust
#: to a larger contaminated fraction, at the cost of tracking curvature
#: less generously.
_VALIDATE_IRLS_PERCENTILE = 25.0
_VALIDATE_IRLS_SCALE = 8.0


def _neighbour_offsets(radius: int) -> np.ndarray:
    """Every ``(dc, dr)`` lattice offset within Chebyshev radius ``radius``,
    excluding ``(0, 0)``."""
    r = np.arange(-radius, radius + 1)
    dc, dr = np.meshgrid(r, r)
    keep = (dc != 0) | (dr != 0)
    return np.stack([dc[keep], dr[keep]], axis=1)


def _fit_local_homographies(
    src: np.ndarray, dst: np.ndarray, weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batched normalised-DLT homography fit, one per row of ``src``/``dst``.

    :param src: ``(B, K, 2)`` source (lattice-offset) points.
    :param dst: ``(B, K, 2)`` destination (image) points.
    :param weight: ``(B, K)`` non-negative weights; 0 excludes a point.
    :return: ``(H (B, 3, 3), ok (B,))`` -- ``ok`` False where the fit is
        under-determined (fewer than 4 effectively-weighted, non-collinear
        points).
    """
    weight_sum = weight.sum(1, keepdims=True)
    weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    normalised = []
    transforms = []
    for points in (src, dst):
        centre = (points * weight[..., None]).sum(1) / weight_sum
        spread = (np.sqrt(((points - centre[:, None, :]) ** 2).sum(-1)) * weight).sum(1) / weight_sum[:, 0]
        scale = np.sqrt(2) / np.maximum(spread, 1e-12)
        transform = np.zeros((len(points), 3, 3))
        transform[:, 0, 0] = scale
        transform[:, 1, 1] = scale
        transform[:, 0, 2] = -scale * centre[:, 0]
        transform[:, 1, 2] = -scale * centre[:, 1]
        transform[:, 2, 2] = 1
        transforms.append(transform)
        normalised.append(np.stack([
            (points[..., 0] - centre[:, None, 0]) * scale[:, None],
            (points[..., 1] - centre[:, None, 1]) * scale[:, None],
            np.ones(points.shape[:2]),
        ], axis=-1))
    s, d = normalised
    zero = np.zeros_like(s)
    row1 = np.concatenate([zero, -s, d[..., 1:2] * s], axis=-1)
    row2 = np.concatenate([s, zero, -d[..., 0:1] * s], axis=-1)
    a = np.concatenate([row1, row2], axis=1) * np.concatenate([weight, weight], axis=1)[..., None]
    gram = np.einsum("bki,bkj->bij", a, a)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    h = eigenvectors[:, :, 0].reshape(-1, 3, 3)
    ok = eigenvalues[:, 1] > 1e-10 * np.maximum(eigenvalues[:, -1], 1e-300)
    homography = np.linalg.inv(transforms[1]) @ h @ transforms[0]
    scale = homography[:, 2:3, 2:3]
    homography = homography / np.where(np.abs(scale) > 1e-15, scale, 1.0)
    return homography, ok


def _apply_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``homography`` (B, 3, 3) applied to ``points`` (B, K, 2) -> (B, K, 2).

    An ill-conditioned local homography (near-collinear or otherwise
    degenerate support) can carry a near-zero homogeneous coordinate for a
    projected point; that is a legitimate "this fit cannot be trusted"
    signal, not a crash or a leaked warning, so the division is explicit
    about ignoring it -- callers are expected to treat a non-finite result
    as unverifiable rather than as a real prediction.
    """
    ones = np.ones(points.shape[:2] + (1,))
    projected = np.einsum("bij,bkj->bki", homography, np.concatenate([points, ones], axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return projected[..., :2] / projected[..., 2:3]


def validate_grid_board_corners(
    lattice_cr: np.ndarray, image_points: np.ndarray,
    radius: int = _VALIDATE_RADIUS, tau_rel: float = _VALIDATE_TAU_REL,
    tau_abs: float = _VALIDATE_TAU_ABS, irls_iterations: int = _VALIDATE_IRLS_ITERATIONS,
    min_neighbours: int = _VALIDATE_MIN_NEIGHBOURS,
) -> np.ndarray:
    """
    Leave-one-out lattice-consistency check: keep a corner only where its
    lattice neighbours predict it.

    For each corner, fit a homography from its Chebyshev-radius-``radius``
    lattice neighbours (excluding the corner itself) to their image
    positions, Tukey-reweighted over ``irls_iterations`` rounds so a cluster
    of wrong neighbours cannot drag the fit, and compare the corner with the
    homography's prediction. A corner is removed when the residual exceeds
    ``max(tau_abs, tau_rel * local_square_px) + model_uncertainty``, where
    ``model_uncertainty`` is a leave-one-out estimate of how much the local
    homography itself can be trusted at exactly this corner (see the
    "model uncertainty" note below) -- not just how far the corner sits from
    a single fit. Removal is greedy, worst offender first, rescoring the
    neighbourhood after each removal, then one pass that reinstates a
    removed corner if it now agrees with the cleaned neighbourhood (recovers
    a corner only *dragged* down by a genuinely wrong one nearby). A corner
    with fewer than ``min_neighbours`` non-collinear neighbours cannot be
    checked at all, and is dropped rather than trusted; so is one whose
    non-finite position makes it impossible to check in the first place.

    Model uncertainty: a plain residual-vs-threshold test assumes the local
    homography is itself a trustworthy little model of the lattice near the
    corner -- true in the interior, where neighbours surround the corner on
    every side, but not at the edge of a large, strongly distorted, steeply
    tilted board, where the neighbourhood only has neighbours on one side
    and the fit is already *extrapolating* to reach the corner at all. Such
    a fit can look confident (low in-sample residual on its own neighbours)
    while being a poor predictor of the untested point. This is measured
    directly: refit the same (Tukey-reweighted) homography once per
    neighbour, each time leaving that one neighbour out, and see how far the
    prediction at the corner's own lattice offset -- never itself part of
    any of these fits -- moves across the refits. An interior corner's fit
    barely reacts to dropping any single neighbour (near-zero spread); an
    extrapolating edge corner's fit can swing a long way (large spread).
    Adding that spread to the tolerance lets a corner survive when the
    *model* is what is uncertain there, without loosening the tolerance
    everywhere else.

    Validated in stage A task A3 against exact distorted lattices (no false
    removals up to a wide-lens model-error floor of 0.098 squares), the v1
    silent-corruption ghost cases (all 3 wrong corners removed, 0 false
    removals), injected 10 px shifts/label swaps/an L of 3 shifted corners
    (99%+ caught), and against combinations well beyond that envelope (large
    boards, k1 to -0.30, tilt to 75 degrees) that the model-uncertainty term
    above was added to cover -- see
    ``tests/test_aruco2_gridboard_quality.py``.

    :param lattice_cr: ``(N, 2)`` ``(col, row)`` lattice coordinates.
    :param image_points: ``(N, 2)`` image positions (post-refinement). A
        non-finite position is dropped up front rather than crashing or
        being treated as verified.
    :return: an ``(N,)`` bool keep mask.
    """
    lattice_cr = np.asarray(lattice_cr, dtype=np.int64).reshape(-1, 2)
    pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    n = len(lattice_cr)
    if n == 0:
        return np.zeros(0, dtype=bool)
    offsets = _neighbour_offsets(radius)
    # The immediate (Chebyshev radius 1) ring of ``offsets``, used for the
    # smaller-support model-uncertainty check inside ``score`` below.
    near_mask = np.max(np.abs(offsets), axis=1) <= 1
    # A non-finite corner can neither be checked nor trusted as anyone
    # else's neighbour; starting it out of ``alive`` keeps it (and only it)
    # out of every homography fit below instead of poisoning them with NaN.
    alive = np.isfinite(pts).all(axis=1)
    grid_h = int(lattice_cr[:, 1].max()) + 1
    grid_w = int(lattice_cr[:, 0].max()) + 1
    grid = -np.ones((grid_h, grid_w), dtype=np.int64)
    grid[lattice_cr[:, 1], lattice_cr[:, 0]] = np.arange(n)

    def score(
        indices: np.ndarray, alive_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Residual, local square size, model uncertainty and verifiability
        for ``indices`` (see the function docstring)."""
        b = len(indices)
        neighbour_cr = lattice_cr[indices][:, None, :] + offsets[None]
        in_bounds = ((neighbour_cr[..., 0] >= 0) & (neighbour_cr[..., 1] >= 0)
                     & (neighbour_cr[..., 0] < grid_w) & (neighbour_cr[..., 1] < grid_h))
        clipped = np.stack(
            [np.clip(neighbour_cr[..., 0], 0, grid_w - 1), np.clip(neighbour_cr[..., 1], 0, grid_h - 1)],
            axis=-1)
        neighbour_idx = np.where(in_bounds, grid[clipped[..., 1], clipped[..., 0]], -1)
        present = neighbour_idx >= 0
        present &= np.where(present, alive_mask[np.maximum(neighbour_idx, 0)], False)
        weight = present.astype(np.float64)
        src = neighbour_cr.astype(np.float64) - lattice_cr[indices][:, None, :]
        dst = np.where(present[..., None], pts[np.maximum(neighbour_idx, 0)], 0.0)

        def support(w: np.ndarray, min_count: int = min_neighbours) -> np.ndarray:
            count = (w > 0).sum(1)
            mean = np.where(w[..., None] > 0, src, 0).sum(1) / np.maximum(count, 1)[:, None]
            centred = np.where(w[..., None] > 0, src - mean[:, None, :], 0)
            covariance = np.einsum("bki,bkj->bij", centred, centred) / np.maximum(count, 1)[:, None, None]
            smallest_eigenvalue = np.linalg.eigvalsh(covariance)[:, 0]
            # >= min_count AND spread over both axes (not collinear).
            return (count >= min_count) & (smallest_eigenvalue > 0.1)

        verifiable = support(weight)
        # dummy destination for absent points, so normalisation stays finite
        fallback = (dst * weight[..., None]).sum(1) / np.maximum(weight.sum(1), 1)[:, None]
        dst = np.where(present[..., None], dst, fallback[:, None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            homography, fit_ok = _fit_local_homographies(src, dst, weight)
        verifiable &= fit_ok

        unit = np.array([[[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]], dtype=np.float64).repeat(b, 0)

        def predict(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            projected = _apply_homography(h, unit)
            valid = np.all(np.isfinite(projected), axis=(1, 2))
            with np.errstate(invalid="ignore"):
                spacing = np.linalg.norm(projected[:, 1:] - projected[:, :1], axis=-1).mean(1)
            square_px = np.where(valid, spacing, np.nan)
            return projected[:, 0], square_px

        prediction, square = predict(homography)
        verifiable &= np.isfinite(square)
        for _ in range(irls_iterations):
            with np.errstate(divide="ignore", invalid="ignore"):
                residual_n = np.linalg.norm(_apply_homography(homography, src) - dst, axis=-1)
            safe_square = np.nan_to_num(square, nan=0.0)
            base_threshold = np.maximum(tau_abs, tau_rel * safe_square)
            # Looser than the final tau_abs/tau_rel comparison on purpose,
            # and adaptively so -- see _VALIDATE_IRLS_SCALE.
            present_residual = np.where(present, residual_n, np.nan)
            with warnings.catch_warnings():
                # "All-NaN slice" when a corner has no present neighbours at
                # all (already unverifiable regardless); not a real warning.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                robust_residual = np.nanpercentile(
                    present_residual, _VALIDATE_IRLS_PERCENTILE, axis=1)
            robust_residual = np.nan_to_num(robust_residual, nan=0.0, posinf=0.0)
            threshold_n = np.maximum(
                base_threshold, _VALIDATE_IRLS_SCALE * robust_residual)[:, None]
            with np.errstate(over="ignore", invalid="ignore"):
                u = np.nan_to_num(residual_n, nan=np.inf) / (2.0 * threshold_n)
                # u is only meaningful (and only used) where u < 1; a huge or
                # infinite u squares to overflow harmlessly here, since the
                # mask below discards it regardless.
                tukey_w = np.where((u < 1) & present, (1 - u ** 2) ** 2, 0.0)
            refit_ok = support(tukey_w)
            with np.errstate(divide="ignore", invalid="ignore"):
                homography_r, fit_ok_r = _fit_local_homographies(src, dst, np.sqrt(tukey_w))
            update = refit_ok & fit_ok_r & np.all(np.isfinite(homography_r), axis=(1, 2))
            homography = np.where(update[:, None, None], homography_r, homography)
            prediction, square = predict(homography)
            verifiable &= np.isfinite(square)

        # Model uncertainty (see the function docstring): refit using only
        # the immediate (Chebyshev radius 1) ring of neighbours -- a
        # smaller-support, lower-order check -- and compare its prediction
        # at the corner's own lattice offset with the full-neighbourhood
        # fit's. In the interior the two agree closely (the lattice is
        # locally planar at either scale). At an edge, where the far fit is
        # already extrapolating across several squares of curvature to
        # reach a corner with support on one side only, shrinking the
        # support changes the extrapolation distance and reveals how much
        # the far prediction owes to curvature the model cannot represent,
        # rather than to the corner's own position. ``_MODEL_ERROR_SCALE``
        # corrects the near/far gap's own systematic undershoot of the true
        # bias (near still extrapolates too, just less): calibrated in
        # tests/test_aruco2_gridboard_quality.py against the exact
        # parameter range this exists for.
        near_weight = tukey_w * near_mask[None, :]
        near_ok = support(near_weight, min_count=4)
        with np.errstate(divide="ignore", invalid="ignore"):
            homography_near, fit_near_ok = _fit_local_homographies(src, dst, near_weight)
            prediction_near = _apply_homography(
                homography_near, np.zeros((b, 1, 2)))[:, 0, :]
        near_ok &= fit_near_ok & np.all(np.isfinite(prediction_near), axis=1)
        with np.errstate(invalid="ignore"):
            disagreement = np.linalg.norm(prediction_near - prediction, axis=1)
        model_uncertainty = np.where(near_ok, _VALIDATE_MODEL_ERROR_SCALE * disagreement, 0.0)

        residual = np.linalg.norm(prediction - pts[indices], axis=1)
        return (
            np.where(verifiable, residual, np.nan),
            np.where(verifiable, square, np.nan),
            verifiable,
            np.where(verifiable, model_uncertainty, np.nan),
        )

    def threshold_of(square: np.ndarray, model_uncertainty: np.ndarray) -> np.ndarray:
        return (np.maximum(tau_abs, tau_rel * np.nan_to_num(square))
                + np.nan_to_num(model_uncertainty))

    residual, square, verifiable, uncertainty = score(np.arange(n), alive)
    removed_log = 0
    while True:
        threshold = threshold_of(square, uncertainty)
        z_score = np.where(alive & verifiable, residual / threshold, -np.inf)
        worst = int(np.argmax(z_score))
        if not np.isfinite(z_score[worst]) or z_score[worst] <= 1.0:
            break
        alive[worst] = False
        removed_log += 1
        neighbours = lattice_cr[worst][None] + offsets
        in_bounds = ((neighbours[:, 0] >= 0) & (neighbours[:, 1] >= 0)
                     & (neighbours[:, 0] < grid_w) & (neighbours[:, 1] < grid_h))
        to_rescore = grid[neighbours[in_bounds, 1], neighbours[in_bounds, 0]]
        to_rescore = to_rescore[(to_rescore >= 0) & alive[np.maximum(to_rescore, 0)]]
        if len(to_rescore):
            residual[to_rescore], square[to_rescore], verifiable[to_rescore], uncertainty[to_rescore] = (
                score(to_rescore, alive))

    readmitted = 0
    if removed_log:
        removed_idx = np.flatnonzero(~alive)
        r2, s2, v2, u2 = score(removed_idx, alive)
        threshold2 = threshold_of(s2, u2)
        back = v2 & (r2 <= threshold2)
        alive[removed_idx[back]] = True
        residual[removed_idx], square[removed_idx], verifiable[removed_idx], uncertainty[removed_idx] = (
            r2, s2, v2, u2)
        readmitted = int(back.sum())

    # A corner that is still alive but cannot be checked (too few
    # non-collinear neighbours, or too few successful leave-one-out refits
    # to estimate a model uncertainty, even after the neighbourhood
    # settled) is dropped, not kept on trust.
    unsupported = np.flatnonzero(alive)
    _, _, final_verifiable, _ = score(unsupported, alive)
    dropped_unsupported = int((~final_verifiable).sum())
    alive[unsupported[~final_verifiable]] = False

    _LOG.debug(
        "aruco2 grid-board validation: %d corner(s) removed, %d readmitted, "
        "%d dropped for having fewer than %d non-collinear neighbours; %d kept.",
        removed_log, readmitted, dropped_unsupported, min_neighbours, int(alive.sum()))
    return alive


def detect_grid_board_corners(
    image, grid_size: tuple[int, int], dict_int: int, marker_size: float,
    ids=None, *, refine: bool = True, validate: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Detect a ChArUco2 grid board and map it to pyCamSet's stable corner ids.

    :param image: the image to detect in, uint8 or an integral 0..255 float.
    :param grid_size: ``(num_squares_x, num_squares_y)``, in markers.
    :param dict_int: the aruco2 dictionary id every square is printed with.
    :param marker_size: the physical pitch between two adjacent markers'
        object points (the printed square size), in the same units
        :data:`ChArUco2.point_data` is stored in.
    :param ids: the marker id on each square, row-major -- exactly
        ``num_squares_x * num_squares_y`` of them -- or None for aruco2's
        default ``0 .. W*H-1``. A board printed with other ids is not found
        under the defaults, so one with its own id range (a cube face) must
        pass them.
    :param refine: run :func:`refine_grid_board_corners` on aruco2's own
        corners before returning them. On by default; ChArUco2 and Ccube2
        use the default.
    :param validate: run :func:`validate_grid_board_corners` afterwards,
        dropping corners its lattice neighbours disagree with. On by
        default; ChArUco2 and Ccube2 use the default.
    :return: ``(corner_ids, image_points)`` -- ``corner_ids`` a 1D int array
        indexing :data:`ChArUco2.point_data`'s rows, ``image_points`` an
        ``(N, 2)`` float array of where each one was found -- or
        ``(None, None)`` when nothing was detected. A corner within
        ``_edge_margin()`` pixels of the image edge is left out, and so is a
        board aruco2 fails on (which is logged as a warning), so a board cut
        by the image edge costs its edge corners, not the image.

        **Pixel convention:** a returned point is in the same pixel-*corner*
        convention as pyCamSet's OpenCV-backed ``ChArUco``/``Ccube``
        (``cv2.aruco.CharucoDetector``) -- 0.5 px from aruco2's own,
        pixel-*centre*, convention on both axes. ChArUco/Ccube need no
        explicit shift for this, under EITHER detector: aruco1 reads them
        with ``CharucoDetector.detectBoard`` directly, and the aruco2 marker
        path is an adapter that hands its own markers to that SAME native
        call (see ``pyCamSet.calibration_targets.markers.aruco2
        .interpolate_board_corners``), so both inherit OpenCV's own
        pixel-corner convention rather than applying a shift of their own.
        ChArUco2/Ccube2 have no such call to inherit it from -- aruco2's
        grid-board detector never goes through ``CharucoDetector`` -- so the
        +0.5 applied just below exists to land their corners on that same
        convention explicitly. This is a deliberate choice, checked by rendering a
        pixel-aligned ``ChArUco`` board and a pixel-aligned ``ChArUco2``
        board with matching analytic corner positions and confirming
        ``ChArUco.find_in_image``/``ChArUco2.find_in_image`` agree to
        ~0.05 px (``tests/test_aruco2_gridboard_quality.py``), so that every
        ArUco-based target in pyCamSet shares one convention and calibration
        results do not depend on which target read the image. The shift is
        applied last, after refinement and validation.
    :raises ValueError: when ``ids`` is given and is not one distinct
        marker of the dictionary per square.
    """
    _require_aruco2()
    board_ids = _grid_board_ids(grid_size, ids, dict_int)
    img = _as_uint8_image(image)
    grid_w, grid_h = int(grid_size[0]), int(grid_size[1])

    # A board cut by the image edge would otherwise make aruco2 raise from
    # cv2.cornerSubPix: see _grid_board_padding. The white border is taken
    # off the returned image points again below.
    pad = _grid_board_padding(img.shape[1])
    padded = cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
        value=(_PAD_VALUE,) * (img.shape[2] if img.ndim == 3 else 1))
    try:
        if board_ids is None:
            found, board = aruco2.detect_grid_board(
                padded, (grid_w, grid_h), int(dict_int))
        else:
            # ids goes by position: it is the binding's fourth argument.
            found, board = aruco2.detect_grid_board(
                padded, (grid_w, grid_h), int(dict_int), board_ids)
    except (cv2.error, ValueError) as err:
        # aruco2 hands OpenCV's cornerSubPix "starting point outside the
        # image" assertion back as a ValueError (see
        # _is_cornersubpix_edge_failure): the padding above is meant to make
        # it unreachable, and if it is still reached, this board is lost
        # from this image, not the image or the folder it is in. Any other
        # ValueError/cv2.error -- a real bug, a malformed board -- is not
        # this failure and must propagate rather than being read as "no
        # detection".
        if not _is_cornersubpix_edge_failure(err):
            raise
        _LOG.warning(
            "aruco2 failed while reading a %dx%d grid board in a %dx%d image, "
            "so no corners are returned for it: %s",
            grid_w, grid_h, img.shape[1], img.shape[0],
            str(err).strip().splitlines()[-1] if str(err).strip() else type(err).__name__)
        return None, None
    if not found or not board.markers:
        return None, None

    obj_points, img_points = aruco2.get_solve_pnp_points(
        board, marker_size=float(marker_size))
    obj_points = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
    img_points = np.asarray(img_points, dtype=np.float64).reshape(-1, 2) - pad
    if obj_points.shape[0] == 0:
        return None, None

    # aruco2 should never hand back a non-finite corner in practice, but
    # refine_grid_board_corners/validate_grid_board_corners are not the
    # place to find out: a NaN here would otherwise reach them (and, worse,
    # poison another corner's neighbourhood-based checks). Dropped up front,
    # explicitly, rather than relying on it happening to fail the
    # edge-margin comparisons below (NaN compares False either way, but
    # that is incidental, not a documented contract).
    finite = np.isfinite(img_points).all(axis=1)
    if not finite.all():
        _LOG.debug(
            "aruco2 grid-board detection: %d corner(s) had a non-finite "
            "position and were dropped before refinement/validation.",
            int((~finite).sum()))
        obj_points = obj_points[finite]
        img_points = img_points[finite]
        if obj_points.shape[0] == 0:
            return None, None

    # Every returned object point lies exactly on the (grid_w+1) x
    # (grid_h+1) intersection lattice at (col, row) * marker_size; recover
    # the row-major index (== the corner's global id in aruco2's own
    # numbering, which pyCamSet.point_data is built to match) by rounding
    # rather than trusting float equality.
    col = np.round(obj_points[:, 0] / float(marker_size)).astype(np.int64)
    row = np.round(obj_points[:, 1] / float(marker_size)).astype(np.int64)
    corner_ids = row * (grid_w + 1) + col

    # A corner this close to the image edge was refined over a window that
    # reaches past it, into the padding (or, unpadded, into cornerSubPix's
    # replicated border), so it is pulled off its true position: by 1-4 px
    # at the median on rendered boards cut by the edge, against 0.2-0.3 px
    # further in. Such a corner is not measured, so it is not returned --
    # before the refinement/validation below, so their bounding-box/lattice
    # work never spends effort on a corner that is being dropped anyway.
    margin = _edge_margin()
    height, width = img.shape[:2]
    inside = ((img_points[:, 0] >= margin) & (img_points[:, 1] >= margin)
              & (img_points[:, 0] <= width - 1 - margin)
              & (img_points[:, 1] <= height - 1 - margin))
    if not np.any(inside):
        return None, None
    corner_ids = corner_ids[inside]
    img_points = img_points[inside].astype(np.float64)

    if refine or validate:
        lattice_cr = _lattice_coords(corner_ids, grid_w)
        if refine:
            refined_points, refined_mask = refine_grid_board_corners(img, lattice_cr, img_points)
            _LOG.debug(
                "aruco2 grid-board refinement: %d of %d corner(s) moved to a saddle point.",
                int(refined_mask.sum()), len(corner_ids))
            img_points = refined_points
        if validate:
            keep = validate_grid_board_corners(lattice_cr, img_points)
            if not np.any(keep):
                return None, None
            corner_ids = corner_ids[keep]
            img_points = img_points[keep]

    if corner_ids.shape[0] == 0:
        return None, None

    # See the docstring's "Pixel convention" note: applied last, so it moves
    # the position refinement/validation actually settled on.
    img_points = img_points + 0.5
    return corner_ids, img_points


def render_grid_board_image(
    grid_size: tuple[int, int], dict_int: int, bit_size: int, ids=None,
) -> np.ndarray:
    """
    aruco2's own raster of a grid board.

    pyCamSet prints from the vector layout in
    :mod:`pyCamSet.calibration_targets.charuco2.layout`; this is the
    reference that layout is checked against, pixel for pixel.

    :param grid_size: ``(num_squares_x, num_squares_y)``, in markers.
    :param dict_int: the aruco2 dictionary every square is printed with.
    :param bit_size: aruco2's scale factor: one marker square, border
        cells included, is ``marker_bits * bit_size`` pixels, so one cell is
        that divided by ``marker_bits + 2``; the band is ``// 4`` of the
        square.
    :param ids: the marker id on each square, row-major, or None for the
        defaults (see :func:`detect_grid_board_corners`).
    :raises ValueError: when ``ids`` is given and is not one distinct
        marker of the dictionary per square.
    """
    _require_aruco2()
    board_ids = _grid_board_ids(grid_size, ids, dict_int)
    grid_w, grid_h = int(grid_size[0]), int(grid_size[1])
    if board_ids is None:
        return aruco2.get_grid_board_image(
            (grid_w, grid_h), int(dict_int), int(bit_size))
    # ids goes by position: it is the binding's fourth argument.
    return aruco2.get_grid_board_image(
        (grid_w, grid_h), int(dict_int), int(bit_size), board_ids)


def dictionary_marker_bits(dict_int: int) -> int:
    """How many bits wide/tall one marker's payload is, for this dictionary."""
    _require_aruco2()
    return int(aruco2.get_predefined_dictionary(int(dict_int)).marker_size)


def dictionary_marker_count(dict_int: int) -> int:
    """
    How many distinct markers this dictionary holds.

    That is the rows of its ``bytes_list`` -- the count aruco2's own board
    renderer checks a board against -- not ``Dictionary.size()``, which
    counts every marker once per rotation.
    """
    _require_aruco2()
    dictionary = aruco2.get_predefined_dictionary(int(dict_int))
    return int(np.asarray(dictionary.bytes_list).shape[0])


def rotation_ambiguous_marker_ids(dict_int: int, ids) -> list[int]:
    """
    The markers among ``ids`` that read the same after a half turn.

    Such a marker's orientation cannot be told from its bits, so aruco2 may
    read it a half turn out, its corners then disagree with its neighbours',
    and aruco2 drops the whole board (seen with DICT_ARUCO_ORIGINAL on a
    32x32 board and on a 5x5 board of ids 999..1023). A marker that
    matches its quarter turn matches its half turn too, so checking the half
    turn finds every one. Of aruco2's dictionaries only DICT_ARUCO_ORIGINAL
    has one today (id 1023); this is computed from the bits rather than
    assumed, so it holds for any dictionary.
    """
    _require_aruco2()
    dictionary = aruco2.get_predefined_dictionary(int(dict_int))
    ambiguous = []
    for marker_id in ids:
        bits = np.asarray(dictionary.get_marker_bits(int(marker_id)))
        if np.array_equal(bits, np.rot90(bits, 2)):
            ambiguous.append(int(marker_id))
    return ambiguous


def refuse_rotation_ambiguous_markers(dict_int: int, ids, dict_name, what: str) -> None:
    """
    Refuse a board that would carry a marker aruco2 cannot orient.

    :param what: the board, as the message should name it.
    :raises ValueError: when any of ``ids`` is rotation-ambiguous.
    """
    ambiguous = rotation_ambiguous_marker_ids(dict_int, ids)
    if ambiguous:
        raise ValueError(
            f"{what} would carry marker(s) {ambiguous[:5]} of {dict_name!r}, "
            f"which read the same after a half turn: aruco2 cannot tell their "
            f"orientation, and a marker read the wrong way round makes it "
            f"drop the whole board. Choose a smaller board or another "
            f"dictionary.")


#: False detections aruco2's grid-board detector is known to make, as
#: ``(dictionary name, host marker id, ghost marker id)``: a patch inside
#: the host square decodes as the ghost marker. When both are on one board
#: the ghost conflicts with the genuine marker and aruco2 drops the whole
#: board -- at some pixel scales only, depending on where the host sits.
_KNOWN_GRID_BOARD_GHOSTS = (
    # Found on 27x27 and larger 4x4 boards with the default ids: the white
    # 3x4-cell patch in square 688 reads as marker 17 at a half turn.
    ("DICT_4X4_1000", 688, 17),
)


def warn_known_false_detections(dict_int: int, ids, dict_name, what: str) -> bool:
    """
    Log a warning when a board carries a host and ghost pair from
    :data:`_KNOWN_GRID_BOARD_GHOSTS`. The board still builds.

    :param ids: the marker ids on one board -- a ghost only matters on the
        board it conflicts with.
    :param what: the board, as the message should name it.
    :return: whether a warning was logged.
    """
    _require_aruco2()
    on_board = {int(i) for i in ids}
    warned = False
    for name, host, ghost in _KNOWN_GRID_BOARD_GHOSTS:
        if int(getattr(aruco2, name)) != int(dict_int):
            continue
        if host in on_board and ghost in on_board:
            _LOG.warning(
                "%s carries markers %d and %d of %r. aruco2 has a known false "
                "detection on such boards: square %d can be read as marker %d, "
                "which at some image scales makes it drop the whole board. A "
                "board of at most %d squares or a larger dictionary (such as "
                "DICT_5X5_1000) avoids it.",
                what, host, ghost, dict_name, host, ghost, host)
            warned = True
    return warned


def grid_board_marker_bits(
    grid_size: tuple[int, int], dict_int: int, ids=None,
) -> np.ndarray:
    """
    The payload bits of every marker on a grid board, True for black.

    ``Dictionary.get_marker_bits`` gives 1 for a *white* cell, in the
    orientation aruco2 draws the marker (rotation 0); this flips it to True
    for black, which is what :func:`pyCamSet.calibration_targets.charuco2
    .layout.grid_board_rectangles` takes. Neither the polarity nor the
    orientation is assumed: both were established by rasterising the layout
    built from these bits and comparing it, pixel for pixel, with
    ``aruco2.get_grid_board_image`` for several dictionaries, board shapes
    and id sets -- a comparison ``tests/test_charuco2_target.py`` keeps
    running.

    :param grid_size: ``(num_squares_x, num_squares_y)``, in markers.
    :param dict_int: the aruco2 dictionary the markers come from.
    :param ids: the marker id on each square, row-major, or None for
        ``0 .. W*H-1``.
    :return: a ``(W*H, m, m)`` bool array, row-major by square.
    :raises ValueError: when ``ids`` is not one distinct marker of the
        dictionary per square.
    """
    _require_aruco2()
    grid_w, grid_h = int(grid_size[0]), int(grid_size[1])
    # The defaults are checked too: a board larger than its dictionary has
    # squares with no marker to print.
    board_ids = _grid_board_ids(
        grid_size, list(range(grid_w * grid_h)) if ids is None else ids, dict_int)
    dictionary = aruco2.get_predefined_dictionary(int(dict_int))
    return np.stack([
        np.asarray(dictionary.get_marker_bits(marker_id)) < 0.5
        for marker_id in board_ids
    ])


class ChArUco2Detector(DetectorParameterisation):
    """
    aruco2's grid-board detector, which takes no settings.

    ``aruco2.detect_grid_board`` is given an image, a board size, a
    dictionary and, optionally, the marker ids, and nothing else -- so there
    is no ``DetectionParameters``-style control here for a form to show or a
    study to sweep. It runs a fixed internal detection pipeline with no
    exposed tuning.
    """

    name = ARUCO2_BACKEND

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return ()

    def unavailable_reason(self, values: dict | None = None) -> str | None:
        if marker_backend_available(ARUCO2_BACKEND):
            return None
        return (
            "ChArUco2 detection requires the 'aruco2' package, which is not "
            "installed. It is not published on PyPI: build it from the "
            "third_party/aruco2 submodule, as described under 'Installing "
            "the aruco2 backend' in pyCamSet's CITATION.md -- unlike "
            "ChArUco/Ccube, ChArUco2 has no ArUco 1 (OpenCV) fallback to "
            "switch to.")


#: Shared rather than built per target: it describes nothing per instance.
CHARUCO2_DETECTOR = ChArUco2Detector()
