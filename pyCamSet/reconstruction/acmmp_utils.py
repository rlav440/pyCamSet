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


def write_pair_file(f: TextIOWrapper, pair_list, scores: np.ndarray | None = None, score_fmt: str = "{}"):
    """
    Given a list of pairs and a file handler, writes that pair list to the file.

    :param f: the handle of the file to write.
    :param pair_list: for each view ``i`` (in order), the candidate source-view
        indices for that view, already in the order they should be written.
    :param scores: an optional ``(N, N)`` matrix of per-pair scores. When
        given, view ``i``'s candidate ``cam_id`` is written with
        ``scores[i, cam_id]`` in place of the original writer's constant
        score of ``1`` -- used by an unbounded, ranked-by-score export such
        as :func:`pyCamSet.utils.saving.camset_to_apde`, where "1" would
        throw away the ranking the caller computed.
    :param score_fmt: format spec applied to each score (e.g. ``"{:.6e}"``
        for scientific notation). Only used when ``scores`` is given.
    """
    f.write(f"{int(len(pair_list))}" + '\n')
    for idi, list_vals in enumerate(pair_list):
        f.write(f"{idi}" + '\n')
        if scores is None:
            line_string = f"{len(list_vals)} "
            line_string += " ".join([f"{cam_id} 1" for cam_id in list_vals])
            f.write(line_string + '\n')
        else:
            terms = " ".join(
                f"{cam_id} {score_fmt.format(scores[idi, cam_id])}" for cam_id in list_vals
            )
            line = f"{len(list_vals)} {terms}" if terms else f"{len(list_vals)}"
            f.write(line + '\n')
    return

def get_v_vec(ext):
    """
    Gets the view vector of a camera given the extrinsic

    :param ext: The extrinsic matrix of a camera.
    """
    return ext[:3,:3] @ np.array([0,0,1])

def calc_pairs(c_vec, r_param: ReconParams, rng=None, pick_closest=False):
    """
    Calclulates the likely pairs from camera view vectors.

    :param c_vec: the camera view vectors.
    :param r_param: the parameters of the reconstruction. Places limits on
        acceptable pairs.
    :param rng: an rng seed for reproducibility.
    :param pick_closest: whether to sort the cameras by angle, or use random selection
    :return pairs: a list of lists of the acceptable pairs for each camera.

    """
    if rng is None:
        rng = np.random.default_rng()
    c_vec /= np.linalg.norm(c_vec, axis=1, keepdims=True)
    t = c_vec[None, ...] * c_vec[:, None]
    ang = np.arccos(np.sum(t, axis=-1)) * 180 / np.pi
    mask = np.logical_and(
        ang > r_param.minangle, ang < r_param.maxangle
    )
    returned_pairs = []
    for idx, masklet in enumerate(mask):
        valid_points = np.where(masklet)[0]
        if len(valid_points) < r_param.max_n_view:
            returned_pairs.append(valid_points)
        else:
            if not pick_closest:
                returned_pairs.append(
                    rng.choice(valid_points, r_param.max_n_view, replace=False)
                )
            else:
                # pick the closest
                dists_sorted = np.argsort(ang[idx, valid_points])
                returned_pairs.append(
                    valid_points[dists_sorted][:r_param.max_n_view]
                )
    return returned_pairs


# ---------------------------------------------------------------------------
# APDe-MVS / ACMMP pair scoring for a calibration rig (no reconstructed scene)
#
# calc_pairs above windows candidates by the angle *between camera view
# vectors* and caps the list at max_n_view -- tuned for a roughly
# forward-facing capture. A calibration rig's cameras converge on a shared
# target instead, so they have *opposing* view directions by construction:
# that angle sits near 180 degrees for the far side of the rig and every
# pair either fails the window or, under the simpler
# baseline * cos(angle) scoring this replaced, clamps to a score of exactly
# zero. The functions below score each pair by the angle it subtends at a
# shared convergence point instead, which stays informative for exactly
# this rig shape, and are unbounded rather than capped -- the caller decides
# how many of the ranked candidates to use.
# ---------------------------------------------------------------------------

def apde_view_geometry(cams) -> tuple[np.ndarray, np.ndarray]:
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


def apde_convergence_point(centres: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, bool]:
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


def calc_apde_pair_scores(cams) -> tuple[np.ndarray, bool]:
    """
    The (N, N) MVSNet-style pairwise score matrix for an APDe-MVS/ACMMP
    ``pair.txt`` export of a calibration rig, in ``cams.get_names()`` order.

    A CameraSet is a calibration, not a reconstruction: there are no scene
    points to compute a real co-visibility or photo-consistency score from.
    This uses a single stand-in scene point ``p`` -- see
    :func:`apde_convergence_point` -- and scores each pair ``(i, j)`` by the
    MVSNet-style piecewise-Gaussian (:func:`mvsnet_pair_score`) of the angle
    subtended at ``p`` between the rays ``p - c_i`` and ``p - c_j``.

    READ THE ORDER, NOT THE MAGNITUDE. The score is peaked at a few degrees
    because that is what MVSNet's heuristic is for: image sets where
    neighbouring views are a short step apart. A calibration rig is
    wide-baseline by construction -- cameras tens of degrees apart, often
    ninety -- so every pair lands far out in the tail and the scores come
    out vanishingly small (e.g. ``~2e-16`` for 90-degree ring neighbours,
    ``~3e-67`` for the opposite view). That ranks correctly, which is what a
    top-k cutoff needs, but the numbers are not meaningful as weights.

    :param cams: pyCamSet CameraSet object
    :return: (scores, well_conditioned) -- the (N, N) score matrix, and
        whether the rig's optical axes converged well enough to define ``p``
        from the well-conditioned lstsq solve (see
        :func:`apde_convergence_point`) rather than its degenerate-axes
        fallback; the caller should warn when this is False, since the
        fallback point is a much rougher stand-in.
    """
    centres, directions = apde_view_geometry(cams)      # (N,3) each, cams.get_names() order
    p, well_conditioned = apde_convergence_point(centres, directions)

    rays = p[np.newaxis, :] - centres                    # (N,3): ray from each camera centre to p
    ray_norms = np.linalg.norm(rays, axis=1, keepdims=True)
    ray_norms = np.where(ray_norms > 1e-12, ray_norms, 1.0)  # guard a camera centre coinciding with p
    unit_rays = rays / ray_norms

    cos_theta = np.clip(unit_rays @ unit_rays.T, -1.0, 1.0)   # (N,N) cosine of the angle at p between ray i, ray j
    theta_deg = np.degrees(np.arccos(cos_theta))
    scores = mvsnet_pair_score(theta_deg)                # (N,N), same convention as the loop below
    return scores, well_conditioned
