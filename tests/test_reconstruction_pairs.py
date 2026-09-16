"""Pair selection for MVSNet/ACMMP-format exports.

``calc_pairs`` and ``CameraSet.write_to_txt``'s ``pair.txt`` had no test
coverage of their own: the exported cameras were checked, the candidate
lists beside them were not.  These tests pin the behaviour that path has
today -- the angle window, the cap, the ordering, and the exact bytes of
the default ``pair.txt`` -- so that merging it with the convergence-point
scoring used by the APDe-MVS export can be shown to preserve it.

The rigs here are fans of cameras rotated about +y, chosen so that every
pairwise view angle inside a row is distinct: a rig with tied angles would
make the closest-first ordering depend on how ``np.argsort`` breaks ties
between two floats that differ only by rounding.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from pyCamSet import Camera, CameraSet
from pyCamSet.reconstruction.acmmp_utils import (
    ReconParams, calc_pair_scores, calc_pairs, normalise_pair_scores,
    pair_angles_from_view_vectors, select_pairs,
)

from conftest import REF_INTRINSIC, REF_RES

# Yaws (degrees about +y) whose pairwise differences are distinct within
# every row, so the closest-first order below is unambiguous.
FAN_YAWS = (0.0, 6.0, 15.0, 27.0, 42.0)


def _view_vectors(yaws_deg) -> np.ndarray:
    """Unit view vectors for cameras yawed about +y from +z."""
    yaws = np.radians(np.asarray(yaws_deg, dtype=float))
    return np.stack([np.sin(yaws), np.zeros_like(yaws), np.cos(yaws)], axis=1)


def _fan_camera(name, yaw_deg, position=(0.0, 0.0, 0.0)) -> Camera:
    """A camera at *position* whose optical axis is yawed *yaw_deg* about +y.

    ``Camera._update_state`` derives ``view`` from
    ``cam_to_world = inv(extrinsic)``, so the rotation is built as the
    cam-to-world one and inverted, rather than assumed symmetric.
    """
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = Rot.from_euler("y", yaw_deg, degrees=True).as_matrix()
    cam_to_world[:3, 3] = np.asarray(position, dtype=float)
    return Camera(
        extrinsic=np.linalg.inv(cam_to_world),
        intrinsic=REF_INTRINSIC.copy(),
        res=list(REF_RES),
        distortion_coefs=np.zeros(5),
        name=name,
    )


@pytest.fixture
def fan_camset() -> CameraSet:
    """Five cameras spread along x, fanned by FAN_YAWS -- a roughly
    forward-facing capture, the shape ``calc_pairs``' angle window is for."""
    cams = {
        f"cam{idx}": _fan_camera(f"cam{idx}", yaw, (0.1 * idx, 0.0, 0.0))
        for idx, yaw in enumerate(FAN_YAWS)
    }
    return CameraSet(camera_dict=cams)


# --------------------------------------------------------------------------
# calc_pairs: the angle window
# --------------------------------------------------------------------------


def test_window_excludes_angles_outside_minangle_and_maxangle():
    """Only pairs whose view vectors differ by an angle strictly inside
    (minangle, maxangle) are candidates."""
    vecs = _view_vectors(FAN_YAWS)
    pairs = calc_pairs(vecs, ReconParams(minangle=10, maxangle=30, max_n_view=99), pick_closest=True)

    # cam0 (0 deg) sees 6, 15, 27, 42 -> only 15 and 27 are inside (10, 30).
    assert sorted(pairs[0]) == [2, 3]
    # cam4 (42 deg) sees 42, 36, 27, 15 -> only 27 and 15 are inside.
    assert sorted(pairs[4]) == [2, 3]


def test_a_camera_is_never_its_own_candidate():
    """The diagonal angle is 0 degrees, which the default window excludes."""
    vecs = _view_vectors(FAN_YAWS)
    pairs = calc_pairs(vecs, ReconParams(max_n_view=99), pick_closest=True)
    for idx, row in enumerate(pairs):
        assert idx not in list(row)


def test_an_empty_row_is_returned_when_no_pair_fits_the_window():
    """A rig whose cameras all point the same way has no candidate at all
    under a window that starts above zero -- the row is empty, not absent."""
    vecs = _view_vectors([0.0, 0.0, 0.0])
    pairs = calc_pairs(vecs, ReconParams(minangle=3, maxangle=45), pick_closest=True)
    assert len(pairs) == 3
    assert all(len(row) == 0 for row in pairs)


# --------------------------------------------------------------------------
# calc_pairs: selection and ordering
# --------------------------------------------------------------------------


def test_candidates_are_ranked_best_first_whether_or_not_the_row_is_capped():
    """Ranking used to be a side effect of capping -- an uncapped row came
    back in ascending index order, because that branch returned
    ``np.where``'s output untouched, leaving a reader that takes the first
    k candidates with the k lowest-numbered cameras. Every row is ranked
    now.

    Above the score's peak angle (5 degrees) ranking by score and ranking
    by closest agree, so the capped row is the same one the old
    closest-first branch produced.
    """
    vecs = _view_vectors(FAN_YAWS)

    # cam2 (15 deg) sees cam1 at 9, cam3 at 12, cam0 at 15, cam4 at 27.
    uncapped = calc_pairs(vecs.copy(), ReconParams(minangle=3, maxangle=45, max_n_view=99), pick_closest=True)
    assert list(uncapped[2]) == [1, 3, 0, 4]

    capped = calc_pairs(vecs.copy(), ReconParams(minangle=3, maxangle=45, max_n_view=3), pick_closest=True)
    assert list(capped[2]) == [1, 3, 0]


def test_calc_pairs_does_not_modify_the_callers_view_vectors():
    """``c_vec /= norm`` normalised the array the caller passed in."""
    vecs = _view_vectors(FAN_YAWS) * 3.0   # deliberately not unit length
    before = vecs.copy()
    calc_pairs(vecs, ReconParams(), pick_closest=True)
    assert np.array_equal(vecs, before)


def test_pair_angles_do_not_warn_about_an_out_of_domain_arccos():
    """A unit vector dotted with itself lands on 1 + 2e-16 often enough to
    matter: unclipped, that made the diagonal NaN and warned on every call
    (which is what kept a camera out of its own candidate list)."""
    vecs = _view_vectors(FAN_YAWS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        angles = pair_angles_from_view_vectors(vecs)
    assert np.all(np.isfinite(angles))
    assert np.allclose(np.diag(angles), 0.0)


def test_pick_closest_caps_the_row_at_max_n_view():
    vecs = _view_vectors(FAN_YAWS)
    pairs = calc_pairs(vecs, ReconParams(minangle=3, maxangle=45, max_n_view=2), pick_closest=True)

    assert all(len(row) == 2 for row in pairs)
    assert list(pairs[2]) == [1, 3]


def test_every_candidate_is_kept_when_fewer_than_max_n_view_fit_the_window():
    vecs = _view_vectors(FAN_YAWS)
    pairs = calc_pairs(vecs, ReconParams(minangle=10, maxangle=30, max_n_view=9), pick_closest=True)

    # Two candidates fit cam0's window, well under max_n_view: both survive,
    # in ascending index order rather than angle order.
    assert list(pairs[0]) == [2, 3]


def test_random_selection_is_reproducible_for_a_seeded_rng():
    vecs = _view_vectors(FAN_YAWS)
    params = ReconParams(minangle=3, maxangle=45, max_n_view=2)

    first = calc_pairs(vecs.copy(), params, rng=np.random.default_rng(0), pick_closest=False)
    second = calc_pairs(vecs.copy(), params, rng=np.random.default_rng(0), pick_closest=False)

    assert [list(row) for row in first] == [list(row) for row in second]


def test_random_selection_draws_distinct_candidates_from_inside_the_window():
    vecs = _view_vectors(FAN_YAWS)
    pairs = calc_pairs(
        vecs, ReconParams(minangle=10, maxangle=30, max_n_view=2),
        rng=np.random.default_rng(7), pick_closest=False,
    )
    for idx, row in enumerate(pairs):
        assert len(set(row)) == len(row)
        assert idx not in list(row)


# --------------------------------------------------------------------------
# CameraSet.write_to_txt: the default pair.txt
# --------------------------------------------------------------------------

# calc_pairs with minangle=3, maxangle=45, max_n_view=3 over FAN_YAWS, best
# first: cam0 sees 6/15/27/42, cam1 sees 6/9/21/36, cam2 sees 15/9/12/27,
# cam3 sees 27/21/12/15, cam4 sees 42/36/27/15.
EXPECTED_DEFAULT_CANDIDATES = [
    [1, 2, 3],
    [0, 2, 3],
    [1, 3, 0],
    [2, 4, 1],
    [3, 2, 1],
]


def _parse_pair_txt(path):
    """[(ids, scores)] per reference view, from a written pair.txt."""
    lines = path.read_text(encoding="utf-8").splitlines()
    n_views = int(lines[0])
    rows = []
    for i in range(n_views):
        parts = lines[2 + 2 * i].split()
        assert int(lines[1 + 2 * i]) == i          # the index line
        rest = parts[1:]
        ids = [int(v) for v in rest[0::2]]
        scores = [float(v) for v in rest[1::2]]
        assert int(parts[0]) == len(ids)           # the declared count
        rows.append((ids, scores))
    return rows


def test_default_pair_file_lists_the_same_candidates_as_before(fan_camset, tmp_path):
    """The merged pipeline must pick the same cameras for a forward-facing
    rig as the angle-windowed selection it replaced."""
    cams_dir = tmp_path / "cams"
    cams_dir.mkdir()
    fan_camset.write_to_txt(cams_dir, ReconParams(minangle=3, maxangle=45, max_n_view=3))

    rows = _parse_pair_txt(tmp_path / "pair.txt")
    assert [ids for ids, _ in rows] == EXPECTED_DEFAULT_CANDIDATES


def test_default_pair_file_scores_the_ranking_instead_of_writing_a_constant(fan_camset, tmp_path):
    """Every candidate used to be written with a score of 1, throwing away
    the ranking the selection had just computed. Each row is now normalised
    to its best candidate, so the numbers carry the order and the best one
    is exactly 1 rather than a value a 32-bit reader rounds to zero."""
    cams_dir = tmp_path / "cams"
    cams_dir.mkdir()
    fan_camset.write_to_txt(cams_dir, ReconParams(minangle=3, maxangle=45, max_n_view=3))

    for ids, scores in _parse_pair_txt(tmp_path / "pair.txt"):
        assert scores[0] == 1.0
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)


def test_default_export_writes_one_cam_file_per_camera(fan_camset, tmp_path):
    cams_dir = tmp_path / "cams"
    cams_dir.mkdir()
    fan_camset.write_to_txt(cams_dir, ReconParams())

    assert sorted(p.name for p in cams_dir.iterdir()) == [
        f"{idx:08d}_cam.txt" for idx in range(len(fan_camset))
    ]


# --------------------------------------------------------------------------
# Choosing between the two angles
# --------------------------------------------------------------------------


@pytest.fixture
def ring_camset() -> CameraSet:
    """Six cameras on a ring, all pointed at the origin -- the converging
    calibration rig, where the angle between view vectors is 60 to 180
    degrees and no window written in those units can admit anything."""
    cams = {}
    for idx in range(6):
        yaw = 60.0 * idx
        angle = np.radians(yaw)
        # on the ring in the xz-plane _fan_camera yaws within, facing inwards:
        # a camera at (sin a, 0, cos a) looks along (-sin a, 0, -cos a),
        # which is a yaw of a + 180.
        position = np.array([np.sin(angle), 0.0, np.cos(angle)])
        cams[f"cam{idx}"] = _fan_camera(f"cam{idx}", yaw + 180.0, position)
    return CameraSet(camera_dict=cams)


def test_auto_scores_a_converging_rig_at_its_convergence_point(ring_camset):
    scoring = calc_pair_scores(ring_camset, ReconParams())
    assert scoring.strategy == "convergence"
    assert scoring.converged
    assert np.allclose(scoring.point, np.zeros(3), atol=1e-9)


def test_auto_scores_a_diverging_fan_by_view_angle(fan_camset):
    """A fan's optical axes have a perfectly well-conditioned least-squares
    closest point; it just sits behind the cameras, where none of them is
    looking. Conditioning alone would pick the wrong strategy here."""
    scoring = calc_pair_scores(fan_camset, ReconParams())
    assert scoring.strategy == "view_angle"
    assert not scoring.converged


def test_a_converging_rig_gets_candidates_instead_of_an_empty_pair_file(ring_camset, tmp_path):
    """The payoff. Scored by view-vector angle, every pair of this rig is 60
    degrees or more apart and falls outside the default window, so pair.txt
    listed no candidate at all for any view -- a file that describes no
    reconstruction."""
    cams_dir = tmp_path / "cams"
    cams_dir.mkdir()
    ring_camset.write_to_txt(cams_dir, ReconParams())

    windowed = calc_pairs(_view_vectors([60.0 * i + 180.0 for i in range(6)]), ReconParams())
    assert all(len(row) == 0 for row in windowed)   # what the old default produced

    rows = _parse_pair_txt(tmp_path / "pair.txt")
    assert all(len(ids) == 5 for ids, _ in rows)
    for ids, scores in rows:
        assert scores[0] == 1.0
        assert scores == sorted(scores, reverse=True)


def test_scoring_can_be_forced_against_the_rig_shape(ring_camset):
    assert calc_pair_scores(ring_camset, ReconParams(), scoring="view_angle").strategy == "view_angle"
    assert calc_pair_scores(ring_camset, ReconParams(), scoring="convergence").strategy == "convergence"
    with pytest.raises(ValueError):
        calc_pair_scores(ring_camset, ReconParams(), scoring="nearest")


# --------------------------------------------------------------------------
# The shared selection and normalisation steps
# --------------------------------------------------------------------------


def test_select_pairs_never_selects_a_view_as_its_own_candidate():
    """Explicitly, rather than relying on a NaN diagonal or on the window
    starting above zero -- both of which a caller can switch off."""
    scores = np.ones((4, 4))
    for row_idx, row in enumerate(select_pairs(scores)):
        assert row_idx not in list(row)


def test_select_pairs_honours_the_mask_and_the_cap():
    scores = np.array([
        [0.0, 0.9, 0.5, 0.7],
        [0.9, 0.0, 0.3, 0.2],
        [0.5, 0.3, 0.0, 0.4],
        [0.7, 0.2, 0.4, 0.0],
    ])
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 1] = False                      # cam1 is not an admissible pair for cam0

    assert list(select_pairs(scores, mask=mask)[0]) == [3, 2]
    assert list(select_pairs(scores, max_n_view=2)[0]) == [1, 3]


def test_normalise_pair_scores_keeps_the_order_and_puts_the_best_at_one():
    scores = np.array([
        [0.0, 2e-16, 3e-67],
        [2e-16, 0.0, 1e-20],
        [3e-67, 1e-20, 0.0],
    ])
    mask = ~np.eye(3, dtype=bool)
    normalised = normalise_pair_scores(scores, mask)

    assert np.max(normalised[0][mask[0]]) == 1.0
    assert np.argsort(-normalised[0]).tolist() == np.argsort(-scores[0]).tolist()


def test_normalise_pair_scores_leaves_a_row_without_candidates_alone():
    """A view whose every pair was masked out has no maximum to divide by;
    the row must come back untouched rather than as NaN."""
    scores = np.zeros((2, 2))
    normalised = normalise_pair_scores(scores, np.zeros((2, 2), dtype=bool))
    assert np.all(np.isfinite(normalised))
    assert np.array_equal(normalised, scores)
