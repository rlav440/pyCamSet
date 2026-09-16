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

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from pyCamSet import Camera, CameraSet
from pyCamSet.reconstruction.acmmp_utils import ReconParams, calc_pairs

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


def test_pick_closest_orders_candidates_by_ascending_angle_only_when_capping():
    """The closest-first ordering is a consequence of the cap, not a
    property of the row: an uncapped row comes back in ascending index
    order, because that branch returns ``np.where``'s output untouched."""
    vecs = _view_vectors(FAN_YAWS)

    # cam2 (15 deg) sees cam1 at 9, cam3 at 12, cam0 at 15, cam4 at 27.
    uncapped = calc_pairs(vecs.copy(), ReconParams(minangle=3, maxangle=45, max_n_view=99), pick_closest=True)
    assert list(uncapped[2]) == [0, 1, 3, 4]

    capped = calc_pairs(vecs.copy(), ReconParams(minangle=3, maxangle=45, max_n_view=3), pick_closest=True)
    assert list(capped[2]) == [1, 3, 0]


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

# calc_pairs with minangle=3, maxangle=45, max_n_view=3 over FAN_YAWS, closest
# first: cam0 sees 6/15/27/42, cam1 sees 6/9/21/36, cam2 sees 15/9/12/27,
# cam3 sees 27/21/12/15, cam4 sees 42/36/27/15.
EXPECTED_DEFAULT_PAIR_TXT = """\
5
0
3 1 1 2 1 3 1
1
3 0 1 2 1 3 1
2
3 1 1 3 1 0 1
3
3 2 1 4 1 1 1
4
3 3 1 2 1 1 1
"""


def test_default_pair_file_is_the_documented_format(fan_camset, tmp_path):
    """One header line, then per view an index line and a candidate line of
    ``count`` followed by ``id score`` pairs -- score a constant 1."""
    cams_dir = tmp_path / "cams"
    cams_dir.mkdir()
    fan_camset.write_to_txt(cams_dir, ReconParams(minangle=3, maxangle=45, max_n_view=3))

    written = (tmp_path / "pair.txt").read_text(encoding="utf-8")
    assert written == EXPECTED_DEFAULT_PAIR_TXT


def test_default_export_writes_one_cam_file_per_camera(fan_camset, tmp_path):
    cams_dir = tmp_path / "cams"
    cams_dir.mkdir()
    fan_camset.write_to_txt(cams_dir, ReconParams())

    assert sorted(p.name for p in cams_dir.iterdir()) == [
        f"{idx:08d}_cam.txt" for idx in range(len(fan_camset))
    ]
