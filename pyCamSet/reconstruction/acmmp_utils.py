from __future__ import annotations
from dataclasses import dataclass
from io import TextIOWrapper
import numpy as np

# make and print a Ccube target
@dataclass
class ReconParams:
    """
    This is a data class that contains the expected parameters for ACMMP/mvsnet.

    """
    def __init__(
            self, mindist=0.1, maxdist=0.8, steps=192, minangle=3, maxangle=45,
            max_n_view=9
    ):
        self.mindist = mindist
        self.maxdist = maxdist
        self.steps = steps
        self.minangle = minangle
        self.maxangle = maxangle
        self.max_n_view = max_n_view


def write_pair_file(f: TextIOWrapper, pair_list, scores: np.ndarray, score_fmt: str = "{:.6e}"):
    """
    Given a list of pairs and a file handler, writes that pair list to the file.

    :param f: the handle of the file to write.
    :param pair_list: for each view ``i`` (in order), the candidate source-view
        indices for that view, already in the order they should be written.
    :param scores: an ``(N, N)`` matrix of per-pair scores; view ``i``'s
        candidate ``cam_id`` is written with ``scores[i, cam_id]``. Normally
        row-normalised by :func:`normalise_pair_scores`, so the best candidate
        for each reference view is written as exactly ``1``. ``pair_list``
        decides how many candidates each view gets; this function does not
        itself cap it.
    :param score_fmt: format spec applied to each score. Scientific notation
        by default: fixed-point printed ``0.000000`` for every pair of a
        wide-baseline rig, which a reader that keeps only ``score > 0`` reads
        as "no candidates at all".
    """
    f.write(f"{int(len(pair_list))}" + '\n')
    for idi, list_vals in enumerate(pair_list):
        terms = " ".join(
            f"{cam_id} {score_fmt.format(scores[idi, cam_id])}" for cam_id in list_vals
        )
        f.write(f"{idi}" + '\n')
        line = f"{len(list_vals)} {terms}" if terms else f"{len(list_vals)}"
        f.write(line + '\n')
    return

def get_v_vec(ext):
    """
    Gets the view vector of a camera given the extrinsic

    :param ext: The extrinsic matrix of a camera.
    """
    return ext[:3,:3] @ np.array([0,0,1])


# ---------------------------------------------------------------------------
# Pair selection for an MVSNet/ACMMP-format ``pair.txt``.
#
# Every pairing here is the same pipeline -- angles -> mask -> score -> rank
# -> cap -> write -- and differs only in where the angle comes from:
#
#   view_angle:  the angle between two cameras' *view vectors*. Cheap, and
#                the right question for a roughly forward-facing capture,
#                where cameras a few degrees apart look a few degrees apart.
#   convergence: the angle the two cameras subtend at the point the rig
#                converges on. The right question for a calibration rig,
#                whose cameras ring a shared target and so have *opposing*
#                view directions by construction -- a view-vector angle near
#                180 degrees for the far side of the rig, which falls outside
#                any sane window and leaves pair.txt empty.
#
# Both feed one score (MVSNet's piecewise Gaussian) and one selection step,
# so a candidate list means the same thing whichever angle produced it.
# ---------------------------------------------------------------------------


def pair_angles_from_view_vectors(view_vecs) -> np.ndarray:
    """
    The ``(N, N)`` matrix of angles, in degrees, between camera view vectors.

    :param view_vecs: (N, 3) camera view vectors; normalised here, and not
        modified in place -- this used to normalise the caller's own array.
    :return: (N, N) angles in degrees, zero on the diagonal.
    """
    vecs = np.asarray(view_vecs, dtype=float)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    # A dot product of a unit vector with itself comes back as 1 + 2e-16 often
    # enough to matter: unclipped, arccos returns NaN for the whole diagonal
    # and numpy warns on every call.
    cos_theta = np.clip(vecs @ vecs.T, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def pair_angles_at_point(centres, point) -> np.ndarray:
    """
    The ``(N, N)`` matrix of angles, in degrees, subtended at ``point``
    between the rays from each camera centre to it.

    :param centres: (N, 3) camera centres
    :param point: the 3D point the angles are measured at
    :return: (N, N) angles in degrees, zero on the diagonal
    """
    centres = np.asarray(centres, dtype=float)
    rays = np.asarray(point, dtype=float)[np.newaxis, :] - centres
    ray_norms = np.linalg.norm(rays, axis=1, keepdims=True)
    ray_norms = np.where(ray_norms > 1e-12, ray_norms, 1.0)  # guard a centre coinciding with point
    unit_rays = rays / ray_norms
    cos_theta = np.clip(unit_rays @ unit_rays.T, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def mvsnet_pair_score(theta_deg: np.ndarray, theta0: float = 5.0, sigma1: float = 1.0, sigma2: float = 10.0) -> np.ndarray:
    """
    MVSNet-style (Yao et al., *MVSNet: Depth Inference for Unstructured
    Multi-view Stereo*, ECCV 2018) piecewise-Gaussian score of the angle
    ``theta`` (degrees), subtended at a 3D point, between the rays to two
    cameras. The score peaks at ``theta0`` -- a small but non-degenerate
    triangulation angle -- and falls off on both sides: tightly (``sigma1``)
    towards ``theta = 0``, where triangulation degenerates, and more broadly
    (``sigma2``) towards large ``theta``, where the two views no longer see
    overlapping scene content.

    :param theta_deg: subtended angle(s) in degrees
    :param theta0: peak angle in degrees (MVSNet's usual default: 5)
    :param sigma1: std. dev. for theta <= theta0 (MVSNet's usual default: 1)
    :param sigma2: std. dev. for theta > theta0 (MVSNet's usual default: 10)
    :return: score(s) in (0, 1], same shape as theta_deg
    """
    theta = np.asarray(theta_deg, dtype=float)
    sigma = np.where(theta <= theta0, sigma1, sigma2)
    return np.exp(-0.5 * ((theta - theta0) / sigma) ** 2)


def normalise_pair_scores(scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Scale each reference view's scores so its best candidate scores exactly 1.

    The ranking a reader needs is the order within one reference view's row,
    and the raw MVSNet score is peaked at a few degrees: a calibration rig,
    wide-baseline by construction, lands far out in the tail and scores every
    pair somewhere around ``1e-16``. Written as they are, a 32-bit reader
    flushes the whole row to zero -- and the ACMMP family keeps only
    candidates with ``score > 0``, so the row a correct ranking just produced
    is read as "no candidates". Dividing through by the row maximum keeps the
    order exactly, puts the best candidate at 1, and leaves only the far tail
    of a very wide rig at risk of flushing -- the candidates a cap would drop
    first anyway.

    :param scores: (N, N) score matrix
    :param mask: optional (N, N) boolean matrix of which entries are
        candidates at all; entries outside it are ignored when finding each
        row's maximum.
    :return: a new (N, N) matrix; rows whose maximum is not positive and
        finite (no candidates at all) are returned untouched.
    """
    scores = np.asarray(scores, dtype=float)
    candidates = np.isfinite(scores) if mask is None else (mask & np.isfinite(scores))
    row_max = np.max(np.where(candidates, scores, -np.inf), axis=1)
    scale = np.where(np.isfinite(row_max) & (row_max > 0), row_max, 1.0)
    return scores / scale[:, np.newaxis]


def select_pairs(scores: np.ndarray, mask: np.ndarray | None = None,
                 max_n_view: int | None = None, pick_closest: bool = True,
                 rng=None) -> list[np.ndarray]:
    """
    Turn a score matrix into the candidate list for each reference view.

    This is the one selection step every pairing strategy shares: drop what
    the mask excludes, order what is left, and cap it.

    :param scores: (N, N) per-pair scores; higher is a better candidate.
    :param mask: optional (N, N) boolean matrix of admissible pairs, e.g. an
        angle window. The diagonal and any non-finite score are excluded
        regardless -- a camera is never its own candidate.
    :param max_n_view: keep at most this many candidates per reference view.
        ``None`` keeps every admissible one.
    :param pick_closest: rank each row by score, best first. When ``False``,
        the row keeps ascending index order and any cap is applied by drawing
        from the admissible candidates at random.
    :param rng: numpy generator used for that random draw; a fresh default
        generator when ``None``, so repeated calls differ.
    :return: for each view in order, an array of candidate view indices.
    """
    scores = np.asarray(scores, dtype=float)
    n_views = scores.shape[0]
    admissible = np.ones((n_views, n_views), dtype=bool) if mask is None else np.array(mask, dtype=bool)
    admissible = admissible & np.isfinite(scores)
    np.fill_diagonal(admissible, False)

    selected = []
    for idx, row in enumerate(admissible):
        candidates = np.where(row)[0]
        if pick_closest:
            # stable, so equally-scoring candidates keep ascending index order
            candidates = candidates[np.argsort(-scores[idx, candidates], kind="stable")]
        elif max_n_view is not None and len(candidates) > max_n_view:
            if rng is None:
                rng = np.random.default_rng()
            candidates = rng.choice(candidates, max_n_view, replace=False)
        if max_n_view is not None:
            candidates = candidates[:max_n_view]
        selected.append(candidates)
    return selected


def view_geometry(cams) -> tuple[np.ndarray, np.ndarray]:
    """
    Camera centres and unit viewing directions, in ``cams.get_names()`` order.

    Both come straight off ``Camera.position`` and ``Camera.view``, which
    ``Camera._update_state`` already derives from
    ``cam_to_world = np.linalg.inv(cam.extrinsic)`` -- so no extra extrinsic
    inversion happens here.

    :param cams: pyCamSet CameraSet object
    :return: (centres, directions), each an (N, 3) array ordered like cams.get_names()
    """
    cam_names = cams.get_names()
    centres = np.array([cams[name].position for name in cam_names])       # world-frame camera centres
    directions = np.array([cams[name].view for name in cam_names])        # world-frame viewing directions
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)  # guard against non-unit input
    return centres, directions


def convergence_point(centres: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    The point minimising the sum of squared perpendicular distances to every
    camera's optical axis -- the standard least-squares closest-point-to-N-lines
    solution, used as a stand-in "scene point" for a calibration rig that has
    no reconstructed geometry of its own.

    For camera ``i`` with centre ``c_i`` and unit view direction ``d_i``, the
    axis is the line through ``c_i`` along ``d_i``. The minimising point
    solves the normal-equations system ``A @ p = b`` with::

        A = sum_i (I - d_i d_i^T)
        b = sum_i (I - d_i d_i^T) @ c_i

    ("I - d d^T" projects a vector onto the plane perpendicular to d, so this
    is exactly the least-squares closest point to N lines.)

    For an ordinary inward-facing calibration rig, this is (close to) the
    rig's physical convergence point -- the target volume every camera is
    pointed at -- solved with ``np.linalg.lstsq``. For near-parallel axes
    (e.g. a forward-facing stereo array) the lines barely converge at all,
    and ``A`` becomes singular or numerically unstable; this is detected via
    its singular values (near-zero smallest singular value, or a very large
    condition number) rather than trusted blindly, and a documented fallback
    is used instead: a point in front of the rig's centroid, along the mean
    viewing direction, at the rig's own characteristic scale (the mean
    pairwise camera-centre distance). That is a reasonable stand-in scene
    point for a forward-facing array -- "somewhere out in front of all the
    cameras, roughly where they'd all agree the scene starts" -- and it is
    always finite, never NaN, unlike extrapolating from an ill-conditioned
    solve.

    :param centres: (N, 3) camera centres
    :param directions: (N, 3) unit viewing directions
    :return: (p, well_conditioned) -- the 3-vector point estimate, and
        whether it came from the well-conditioned lstsq solve (True) or the
        degenerate-axes fallback (False).
    """
    eye = np.eye(3)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centres, directions):
        proj = eye - np.outer(d, d)   # projector onto the plane perpendicular to this camera's axis
        A += proj
        b += proj @ c

    # A well-conditioned 3x3 normal-equations matrix has all three singular
    # values comfortably away from zero. Near-parallel axes make the
    # smallest singular value collapse towards zero (or exactly zero, for
    # exactly-parallel axes) -- np.linalg.lstsq would still return *some*
    # point for a singular/ill-conditioned system, but that point can be
    # arbitrarily unstable, so this is checked explicitly rather than trusted.
    singular_values = np.linalg.svd(A, compute_uv=False)
    well_conditioned = (
        singular_values[-1] > 1e-8
        and (singular_values[0] / singular_values[-1]) < 1e8
    )

    if well_conditioned:
        p, *_ = np.linalg.lstsq(A, b, rcond=None)
        return p, True

    # Degenerate fallback: axes do not usefully converge.
    centroid = centres.mean(axis=0)
    mean_dir = directions.mean(axis=0)
    mean_dir_norm = np.linalg.norm(mean_dir)
    # opposing view directions could average to (near) zero; fall back again
    # to a fixed axis rather than divide by ~0.
    mean_dir = mean_dir / mean_dir_norm if mean_dir_norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    if len(centres) > 1:
        pairwise = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
        scale = pairwise[np.triu_indices(len(centres), k=1)].mean()
    else:
        scale = 0.0
    scale = scale if scale > 1e-9 else 1.0   # single camera / coincident centres: an arbitrary unit standoff
    p = centroid + mean_dir * scale
    return p, False


def rig_converges(centres: np.ndarray, directions: np.ndarray, point: np.ndarray,
                  well_conditioned: bool) -> bool:
    """
    Whether ``point`` is a convergence point the cameras actually look at.

    A well-conditioned solve is not enough on its own: a *diverging* fan of
    cameras has a perfectly well-conditioned least-squares closest point too,
    it just sits behind the rig, where no camera is looking. The question
    that distinguishes a calibration rig from a forward-facing capture is
    whether every camera's optical axis points towards the shared point, so
    that is what is asked.

    :param centres: (N, 3) camera centres
    :param directions: (N, 3) unit viewing directions
    :param point: the candidate convergence point
    :param well_conditioned: whether ``point`` came from the lstsq solve
    :return: True when the point is in front of every camera
    """
    if not well_conditioned or len(centres) < 2:
        return False
    to_point = np.asarray(point, dtype=float)[np.newaxis, :] - np.asarray(centres, dtype=float)
    return bool(np.all(np.sum(to_point * np.asarray(directions, dtype=float), axis=1) > 0))


@dataclass
class PairScoring:
    """The scores one pairing strategy produced, and how it got there.

    :param scores: (N, N) per-pair scores, higher is a better candidate.
    :param mask: (N, N) admissible pairs -- an angle window, for the
        view-vector strategy; everything off the diagonal otherwise.
    :param strategy: ``"view_angle"`` or ``"convergence"``, the strategy
        actually used (``"auto"`` resolves to one of them).
    :param converged: whether the rig's optical axes meet at a point every
        camera looks towards.
    :param point: the convergence point used, or None for the view-vector
        strategy.
    """
    scores: np.ndarray
    mask: np.ndarray
    strategy: str
    converged: bool
    point: np.ndarray | None = None


def calc_pair_scores(cams, r_param: ReconParams | None = None, scoring: str = "auto") -> PairScoring:
    """
    Score every pair of cameras as candidates for one another in pair.txt.

    :param cams: pyCamSet CameraSet object.
    :param r_param: reconstruction parameters. Only the angle window
        (``minangle``/``maxangle``) is read, and only by the ``view_angle``
        strategy; ``None`` leaves the window off.
    :param scoring: which angle to score.

        ``"view_angle"``
            the angle between camera view vectors, windowed by
            ``r_param``'s ``minangle``/``maxangle``.
        ``"convergence"``
            the angle subtended at the rig's convergence point. No window:
            the window is expressed in view-vector degrees, where a ring rig
            sits at 90-180 and would be filtered away entirely, and the
            score already penalises both a degenerate triangulation angle
            and a non-overlapping one.
        ``"auto"`` (default)
            ``"convergence"`` when the rig actually converges -- see
            :func:`rig_converges` -- and ``"view_angle"`` otherwise, which
            is the forward-facing capture the window was written for.

    :return: a :class:`PairScoring`.
    """
    if scoring not in ("auto", "view_angle", "convergence"):
        raise ValueError(f"unknown pair scoring strategy {scoring!r}")

    centres, directions = view_geometry(cams)
    point, well_conditioned = convergence_point(centres, directions)
    converged = rig_converges(centres, directions, point, well_conditioned)

    if scoring == "auto":
        scoring = "convergence" if converged else "view_angle"

    n_views = len(centres)
    if scoring == "convergence":
        theta_deg = pair_angles_at_point(centres, point)
        mask = ~np.eye(n_views, dtype=bool)
    else:
        theta_deg = pair_angles_from_view_vectors(directions)
        if r_param is None:
            mask = ~np.eye(n_views, dtype=bool)
        else:
            mask = (theta_deg > r_param.minangle) & (theta_deg < r_param.maxangle)
        point = None

    return PairScoring(
        scores=mvsnet_pair_score(theta_deg), mask=mask, strategy=scoring,
        converged=converged, point=point,
    )


def calc_convergence_pair_scores(cams) -> tuple[np.ndarray, bool]:
    """
    The (N, N) pairwise score matrix for a rig scored at its convergence
    point, in ``cams.get_names()`` order -- :func:`calc_pair_scores` with
    ``scoring="convergence"``, returned in the shape an exporter wants.

    A CameraSet is a calibration, not a reconstruction: there are no scene
    points to compute a real co-visibility or photo-consistency score from.
    This uses a single stand-in scene point ``p`` -- see
    :func:`convergence_point` -- and scores each pair ``(i, j)`` by the
    MVSNet-style piecewise Gaussian (:func:`mvsnet_pair_score`) of the angle
    subtended at ``p`` between the rays ``p - c_i`` and ``p - c_j``.

    READ THE ORDER, NOT THE MAGNITUDE. The raw score is peaked at a few
    degrees because that is what MVSNet's heuristic is for: image sets where
    neighbouring views are a short step apart. A calibration rig is
    wide-baseline by construction -- cameras tens of degrees apart, often
    ninety -- so every pair lands far out in the tail and the raw scores come
    out vanishingly small (e.g. ``~2e-16`` for 90-degree ring neighbours,
    ``~3e-67`` for the opposite view). That ranks correctly, which is what a
    top-k cutoff needs, but the numbers are not meaningful as weights, and
    what reaches pair.txt is row-normalised (:func:`normalise_pair_scores`)
    so the best candidate for each view is written as 1 rather than as a
    number a 32-bit reader rounds to zero.

    :param cams: pyCamSet CameraSet object
    :return: (scores, well_conditioned) -- the (N, N) score matrix, and
        whether the rig's optical axes converged well enough to define ``p``
        from the well-conditioned lstsq solve (see
        :func:`convergence_point`) rather than its degenerate-axes fallback;
        the caller should warn when this is False, since the fallback point
        is a much rougher stand-in.
    """
    centres, directions = view_geometry(cams)
    _, well_conditioned = convergence_point(centres, directions)
    return calc_pair_scores(cams, scoring="convergence").scores, well_conditioned


def calc_pairs(c_vec, r_param: ReconParams, rng=None, pick_closest=False):
    """
    Calculates the likely pairs from camera view vectors.

    The view-vector half of :func:`calc_pair_scores` -- kept for callers that
    have view vectors rather than a CameraSet, and so cannot ask where the
    rig converges. Prefer :func:`calc_pair_scores` with :func:`select_pairs`,
    which can.

    :param c_vec: the camera view vectors. Not modified.
    :param r_param: the parameters of the reconstruction. Places limits on
        acceptable pairs.
    :param rng: an rng seed for reproducibility.
    :param pick_closest: whether to rank the cameras by score, or use random
        selection.
    :return pairs: a list of lists of the acceptable pairs for each camera.

    """
    theta_deg = pair_angles_from_view_vectors(c_vec)
    mask = (theta_deg > r_param.minangle) & (theta_deg < r_param.maxangle)
    return select_pairs(
        mvsnet_pair_score(theta_deg), mask=mask, max_n_view=r_param.max_n_view,
        pick_closest=pick_closest, rng=rng,
    )
