"""The APDe-MVS ("cams/" + pair.txt) exporter in pyCamSet.utils.saving.

pyCamSet had no APD-MVS/APDe-MVS exporter before ``camset_to_apde``.  The
existing COLMAP exporter is asserted against actual written file content
rather than "it ran without raising", so this file does the same: it parses
each written ``*_cam.txt`` and ``pair.txt`` back and checks the numbers, not
just their presence.

These tests are synthetic and need no image corpus, mirroring the
``synthetic_camset``/``make_camera`` idiom in ``test_camera_set.py`` and
``test_saving.py`` -- but with a genuine (non-identity) rotation on each
camera, since a rotation of the identity cannot tell a correct R|t block
from a transposed or inverted one.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from pyCamSet import Camera, CameraSet
from pyCamSet.utils.saving import camset_to_apde

from conftest import REF_INTRINSIC, REF_RES, make_camera


def _lookat_camera(name, position, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """A camera at ``position``, oriented so its view direction points at
    ``target``. Used to build inward-facing (converging) and forward-facing
    (parallel-axis) rigs for the pair.txt scoring tests below, where the
    *rotation* -- not just the translation -- has to put every camera's
    view axis through a specific 3D point.
    """
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, dtype=float)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    # cam_to_world rotation: columns are the camera's local axes (x, y, z)
    # expressed in world coordinates -- see Camera._update_state, where
    # position = cam_to_world @ [0,0,0,1] and view = cam_to_world @ [0,0,1,0].
    cam_to_world_rot = np.stack([right, true_up, forward], axis=1)
    # extrinsic (world-to-cam) is the inverse of cam_to_world; for a pure
    # rotation, inverse == transpose.
    rot_world_to_cam = cam_to_world_rot.T
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot_world_to_cam
    extrinsic[:3, 3] = -rot_world_to_cam @ position
    return Camera(
        extrinsic=extrinsic,
        intrinsic=REF_INTRINSIC.copy(),
        res=list(REF_RES),
        distortion_coefs=np.zeros(5),
        name=name,
    )


def _rotated_camera(name, translation, euler_deg_xyz):
    """A camera with a genuine rotation, so the extrinsic round-trip check
    below cannot pass by accident the way it would for R = I."""
    rot = Rot.from_euler("xyz", euler_deg_xyz, degrees=True).as_matrix()
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot
    extrinsic[:3, 3] = translation
    return Camera(
        extrinsic=extrinsic,
        intrinsic=REF_INTRINSIC.copy(),
        res=list(REF_RES),
        distortion_coefs=np.zeros(5),
        name=name,
    )


@pytest.fixture
def apde_camset():
    """Three cameras spread around the origin with distinct rotations, so
    baseline and viewing-angle differ between every pair (a rig where all
    cameras shared one direction could not distinguish the pair.txt score
    from a plain baseline ranking)."""
    cams = {
        "left": _rotated_camera("left", (-0.1, 0.0, 0.3), (0.0, 20.0, 0.0)),
        "centre": _rotated_camera("centre", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        "right": _rotated_camera("right", (0.1, 0.0, 0.3), (0.0, -20.0, 0.0)),
    }
    return CameraSet(camera_dict=cams)


def _read_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


# --------------------------------------------------------------------------
# Block structure of a single *_cam.txt
# --------------------------------------------------------------------------


def test_cam_txt_block_structure_and_blank_lines(apde_camset, tmp_path):
    """Byte-level shape: header, 4-row block, blank, header, 3-row block,
    blank, one depth line -- nothing more, nothing less."""
    camset_to_apde(apde_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)

    lines = _read_lines(tmp_path / "cams" / "00000000_cam.txt")

    assert len(lines) == 12
    assert lines[0] == "extrinsic"
    assert lines[5] == ""
    assert lines[6] == "intrinsic"
    assert lines[10] == ""
    for i in range(1, 5):
        assert len(lines[i].split()) == 4
    for i in range(7, 10):
        assert len(lines[i].split()) == 3
    assert len(lines[11].split()) == 4


# --------------------------------------------------------------------------
# Extrinsic round trip -- the convention this exporter depends on
# --------------------------------------------------------------------------


def test_extrinsic_round_trips_as_world_to_camera_unmodified(apde_camset, tmp_path):
    """The written R|t must equal cam.extrinsic exactly -- no inversion.

    Evidence for the convention: Camera._update_state sets
    cam.cam_to_world = np.linalg.inv(cam.extrinsic), and
    Camera._calc_projection_matrix forms intrinsic @ extrinsic[:3, :4],
    the standard x = K [R|t] X_world projection -- which only holds if
    [R|t] is world-to-camera. export_rig_config (the existing COLMAP
    exporter) relies on the same thing when it takes cam.extrinsic
    directly as "cam-from-world" without inverting it. If this exporter
    had the convention backwards, the parsed block would match
    cam.cam_to_world instead, and this assertion would fail.
    """
    camset_to_apde(apde_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)

    for idx, name in enumerate(apde_camset.get_names()):
        lines = _read_lines(tmp_path / "cams" / f"{idx:08d}_cam.txt")
        parsed = np.array([[float(v) for v in lines[i].split()] for i in range(1, 5)])

        cam = apde_camset[name]
        assert np.allclose(parsed, cam.extrinsic)
        # "centre" sits at the identity pose, where extrinsic == cam_to_world,
        # so the swapped-convention guard only bites for the two off-centre
        # cameras -- which is exactly why the fixture rotates and translates them.
        if not np.allclose(cam.extrinsic, cam.cam_to_world):
            assert not np.allclose(parsed, cam.cam_to_world)
        assert np.allclose(parsed[3], [0.0, 0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# Intrinsics
# --------------------------------------------------------------------------


def test_intrinsics_match_k(apde_camset, tmp_path):
    camset_to_apde(apde_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)

    for idx, name in enumerate(apde_camset.get_names()):
        lines = _read_lines(tmp_path / "cams" / f"{idx:08d}_cam.txt")
        parsed_k = np.array([[float(v) for v in lines[i].split()] for i in range(7, 10)])
        assert np.allclose(parsed_k, apde_camset[name].intrinsic)


# --------------------------------------------------------------------------
# Depth line
# --------------------------------------------------------------------------


def test_depth_line_matches_requested_range_and_computed_interval(apde_camset, tmp_path):
    depth_min, depth_max, depth_num = 0.2, 1.0, 64
    camset_to_apde(apde_camset, tmp_path, depth_min=depth_min, depth_max=depth_max, depth_num=depth_num)

    lines = _read_lines(tmp_path / "cams" / "00000000_cam.txt")
    got_min, got_interval, got_num, got_max = (float(v) for v in lines[11].split())

    assert got_min == pytest.approx(depth_min)
    assert got_max == pytest.approx(depth_max)
    assert int(got_num) == depth_num
    assert got_interval == pytest.approx((depth_max - depth_min) / depth_num)


# --------------------------------------------------------------------------
# pair.txt
# --------------------------------------------------------------------------


def test_pair_txt_shape_for_n_views(apde_camset, tmp_path):
    camset_to_apde(apde_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)
    lines = _read_lines(tmp_path / "pair.txt")

    n_views = len(apde_camset)
    assert int(lines[0]) == n_views
    assert len(lines) == 1 + 2 * n_views

    for i in range(n_views):
        assert int(lines[1 + 2 * i]) == i

        parts = lines[2 + 2 * i].split()
        n_neighbours = int(parts[0])
        assert n_neighbours == n_views - 1  # every other view is a candidate

        rest = parts[1:]
        assert len(rest) == 2 * n_neighbours
        src_ids = [int(v) for v in rest[0::2]]
        scores = [float(v) for v in rest[1::2]]

        assert sorted(src_ids) == sorted(j for j in range(n_views) if j != i)
        assert scores == sorted(scores, reverse=True)  # best candidate listed first
        assert all(s >= 0.0 for s in scores)  # opposing-view scores are clamped, never negative


# --------------------------------------------------------------------------
# File count and the index -> name map
# --------------------------------------------------------------------------


def test_cam_file_count_equals_camera_count(apde_camset, tmp_path):
    camset_to_apde(apde_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)

    cam_files = sorted((tmp_path / "cams").glob("*_cam.txt"))
    assert len(cam_files) == len(apde_camset)
    assert [f.name for f in cam_files] == [f"{i:08d}_cam.txt" for i in range(len(apde_camset))]


def test_cam_index_map_matches_get_names_order(apde_camset, tmp_path):
    camset_to_apde(apde_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)

    lines = _read_lines(tmp_path / "cam_index_map.txt")
    expected = [f"{i:08d} {name}" for i, name in enumerate(apde_camset.get_names())]
    assert lines == expected


# --------------------------------------------------------------------------
# Distortion warning
# --------------------------------------------------------------------------


def test_distortion_triggers_a_logged_warning(tmp_path, caplog):
    distorted = make_camera("wide", distortion=[0.1, 0.0, 0.0, 0.0, 0.0])
    camset = CameraSet(camera_dict={"wide": distorted})

    with caplog.at_level(logging.WARNING, logger="pyCamSet.utils.saving"):
        camset_to_apde(camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)

    assert "distortion" in caplog.text
    assert "wide" in caplog.text


def test_undistorted_cameras_do_not_warn(apde_camset, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="pyCamSet.utils.saving"):
        camset_to_apde(apde_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)

    assert "distortion" not in caplog.text


# --------------------------------------------------------------------------
# pair.txt scoring for a converging rig (DEFECT 1)
#
# The old score, baseline_ij * max(0, cos(angle between view_i, view_j)),
# is exactly 0.0 for EVERY pair on an ordinary inward-facing multi-camera
# rig: cameras that converge on a common point necessarily have *opposing*
# view directions, so the cosine term is always <= 0 and clamps to zero.
# apde_camset above (+/-20 degree rotations, not a converging rig) never
# exercises this, which is why it did not catch the defect.
# --------------------------------------------------------------------------


def _ring_camera(name, angle_deg, radius=1.0, target=(0.0, 0.0, 0.0)):
    angle = np.radians(angle_deg)
    position = (radius * np.cos(angle), radius * np.sin(angle), 0.0)
    return _lookat_camera(name, position, target=target)


@pytest.fixture
def converging_ring_camset():
    """Four cameras on a ring, 90 degrees apart, all pointed at a common
    centre point -- the ordinary multi-camera calibration rig pyCamSet
    exists for."""
    cams = {
        f"cam{i}": _ring_camera(f"cam{i}", angle_deg)
        for i, angle_deg in enumerate([0.0, 90.0, 180.0, 270.0])
    }
    return CameraSet(camera_dict=cams)


def test_pair_scores_positive_varied_and_ordered_for_converging_rig(converging_ring_camset, tmp_path):
    """CONFIRMED to fail against the old implementation (see below): every
    score on this rig came out exactly 0.000000 under
    ``baseline_ij * max(0, cos(angle between view_i, view_j))``, because
    inward-facing cameras have opposing view directions. The replacement
    score must be positive, vary between pairs, and rank each camera's
    angularly-nearest ring neighbour (90 degrees away) ahead of its
    diametrically opposite camera (180 degrees away).
    """
    camset_to_apde(converging_ring_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)
    lines = _read_lines(tmp_path / "pair.txt")

    n_views = 4
    parsed = {}
    for i in range(n_views):
        parts = lines[2 + 2 * i].split()
        rest = parts[1:]
        src_ids = [int(v) for v in rest[0::2]]
        scores = [float(v) for v in rest[1::2]]
        parsed[i] = dict(zip(src_ids, scores))

    all_scores = [s for d in parsed.values() for s in d.values()]
    # positive: this is exactly what the old cosine-clamped score fails --
    # it produces 0.000000 for every single one of these pairs
    assert all(s > 0.0 for s in all_scores)
    # varied: not a single degenerate constant across all pairs (the scores
    # here span many orders of magnitude, so compare exactly rather than
    # rounding to a fixed number of decimals)
    assert len(set(all_scores)) > 1

    # camera 0's ring neighbours (cameras 1 and 3, 90 degrees away) must
    # both outrank its opposite (camera 2, 180 degrees away)
    assert parsed[0][1] > parsed[0][2]
    assert parsed[0][3] > parsed[0][2]
    # the two neighbours are symmetric about camera 0, so their scores
    # should match
    assert parsed[0][1] == pytest.approx(parsed[0][3], rel=1e-6)


def test_pair_scores_are_exactly_zero_under_the_old_cosine_score(converging_ring_camset):
    """Direct evidence that the converging-ring fixture is what DEFECT 1
    describes: recomputing the OLD score formula by hand (rather than
    reimplementing/reverting the fix under test) confirms every pair on
    this rig scores exactly 0.0 under it, which is what made pair.txt
    degenerate to ascending-index order with no information content.
    """
    from pyCamSet.reconstruction.acmmp_utils import view_geometry

    centres, directions = view_geometry(converging_ring_camset)
    n_views = len(centres)
    for i in range(n_views):
        baselines = np.linalg.norm(centres - centres[i], axis=1)
        cosines = directions @ directions[i]
        old_scores = baselines * np.clip(cosines, 0.0, None)
        others = [j for j in range(n_views) if j != i]
        # float64 cos(90 degrees) is ~6e-17, not bit-exact 0 -- close enough
        # that the field-verified symptom (every score prints "0.000000" at
        # 6 decimal places) still holds, which is what DEFECT 1 reported.
        assert all(old_scores[j] == pytest.approx(0.0, abs=1e-9) for j in others)
        assert all(f"{old_scores[j]:.6f}" == "0.000000" for j in others)


# --------------------------------------------------------------------------
# pair.txt scoring for a degenerate (near-parallel axes) rig (DEFECT 1)
# --------------------------------------------------------------------------


@pytest.fixture
def parallel_stereo_camset():
    """A forward-facing stereo pair: both cameras share one viewing
    direction, so their optical axes never converge and the
    least-squares closest-point-to-N-lines solve is rank-deficient."""
    # Forward = +y for both cameras (not the +z up vector, which would make
    # the right = cross(forward, up) computation in _lookat_camera degenerate).
    cams = {
        "l": _lookat_camera("l", (-0.05, 0.0, 0.0), target=(-0.05, 1.0, 0.0)),
        "r": _lookat_camera("r", (0.05, 0.0, 0.0), target=(0.05, 1.0, 0.0)),
    }
    return CameraSet(camera_dict=cams)


def test_pair_scores_finite_for_degenerate_parallel_rig(parallel_stereo_camset, tmp_path, caplog):
    """Near-parallel camera axes must not produce NaN/inf pair scores, and
    the documented fallback (centroid + mean viewing direction, at the
    rig's own baseline scale) must be used and logged rather than silently
    returning garbage from an ill-conditioned solve."""
    with caplog.at_level(logging.WARNING, logger="pyCamSet.utils.saving"):
        camset_to_apde(parallel_stereo_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)

    lines = _read_lines(tmp_path / "pair.txt")
    assert int(lines[0]) == 2
    for i in range(2):
        parts = lines[2 + 2 * i].split()
        scores = [float(v) for v in parts[1:][1::2]]
        assert len(scores) == 1
        assert all(np.isfinite(s) for s in scores)
        assert all(s > 0.0 for s in scores)

    assert "parallel" in caplog.text.lower()
    assert "camset_to_apde: camera axes are near-parallel" in caplog.text
    assert "falling back" in caplog.text.lower()


# --------------------------------------------------------------------------
# Re-export must not leave stale cams/ files behind (DEFECT 2)
# --------------------------------------------------------------------------


def test_reexport_with_fewer_cameras_removes_stale_cam_files(apde_camset, tmp_path):
    """Exporting 3 cameras then 1 into the same output dir must leave
    exactly 1 *_cam.txt file (and a cam_index_map.txt describing only that
    1 camera), not 3 -- and must not touch an unrelated file placed in
    cams/ by something else."""
    camset_to_apde(apde_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)
    cams_dir = tmp_path / "cams"
    assert sorted(f.name for f in cams_dir.glob("*_cam.txt")) == [
        "00000000_cam.txt", "00000001_cam.txt", "00000002_cam.txt",
    ]

    unrelated = cams_dir / "notes.txt"
    unrelated.write_text("do not delete me", encoding="utf-8")

    smaller_camset = CameraSet(camera_dict={"left": apde_camset["left"]})
    camset_to_apde(smaller_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)

    cam_files = sorted(f.name for f in cams_dir.glob("*_cam.txt"))
    assert cam_files == ["00000000_cam.txt"]

    # nothing else in cams/ was touched
    assert unrelated.exists()
    assert unrelated.read_text(encoding="utf-8") == "do not delete me"

    # cam_index_map.txt and pair.txt describe only the new, smaller export
    index_lines = (tmp_path / "cam_index_map.txt").read_text(encoding="utf-8").splitlines()
    assert index_lines == ["00000000 left"]
    pair_lines = _read_lines(tmp_path / "pair.txt")
    assert int(pair_lines[0]) == 1


# --------------------------------------------------------------------------
# pair.txt source-view cap (DEFECT 3)
#
# APD-MVS and APDe-MVS (github.com/whoiszzj/APD-MVS,
# github.com/whoiszzj/APDe-MVS) both `#define MAX_IMAGES 32` and abort
# (exit(EXIT_FAILURE), "Can't process so much images") rather than truncate
# once a reference view's loaded image count -- one reference plus every
# pair.txt candidate scoring above 0 -- exceeds it. Their pair.txt reader
# itself applies no top-k cutoff of its own: it keeps every candidate with
# score > 0, in file order. camset_to_apde must therefore cap the candidate
# list itself, at export time, rather than writing every other camera
# unbounded the way it did before this cap existed.
# --------------------------------------------------------------------------


@pytest.fixture
def large_ring_camset():
    """40 cameras on a ring, pointed at a common centre -- enough views that
    "every other camera" (39 candidates) exceeds APD-MVS/APDe-MVS's 31-source
    limit, so a correct exporter must cap the candidate list rather than
    write all 39."""
    n = 40
    cams = {
        f"cam{i}": _ring_camera(f"cam{i}", angle_deg=360.0 * i / n)
        for i in range(n)
    }
    return CameraSet(camera_dict=cams)


def _parse_pair_row(lines, i):
    """(n_neighbours, [source_ids], [scores]) for reference view i, from an
    already-read pair.txt -- shared by the cap tests below to keep the
    header-offset/stride/interleaving assumptions in one place."""
    parts = lines[2 + 2 * i].split()
    rest = parts[1:]
    return int(parts[0]), [int(v) for v in rest[0::2]], [float(v) for v in rest[1::2]]


def _top_k_ranking(scores, i, n_views, k):
    """The ids expected for reference view i under a correct top-k-by-score
    cap, recomputed independently of the exporter for comparison."""
    full_ranking = sorted((j for j in range(n_views) if j != i), key=lambda j: scores[i, j], reverse=True)
    return full_ranking[:k]


def test_pair_txt_caps_candidates_at_the_apde_mvs_limit(large_ring_camset, tmp_path):
    from pyCamSet.utils.saving import _APDE_MVS_MAX_SRC_VIEWS

    camset_to_apde(large_ring_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)
    lines = _read_lines(tmp_path / "pair.txt")

    n_views = len(large_ring_camset)
    assert n_views - 1 > _APDE_MVS_MAX_SRC_VIEWS  # the fixture must actually exercise the cap

    for i in range(n_views):
        n_neighbours, _, scores = _parse_pair_row(lines, i)
        assert n_neighbours == _APDE_MVS_MAX_SRC_VIEWS
        assert len(scores) == _APDE_MVS_MAX_SRC_VIEWS
        assert scores == sorted(scores, reverse=True)


def test_pair_txt_cap_keeps_the_highest_scoring_candidates(large_ring_camset, tmp_path):
    """Not just the right *count* -- capping must drop the worst-scoring
    candidates, not an arbitrary 8 out of 39 (e.g. not the last 8 written)."""
    from pyCamSet.reconstruction.acmmp_utils import calc_convergence_pair_scores
    from pyCamSet.utils.saving import _APDE_MVS_MAX_SRC_VIEWS

    camset_to_apde(large_ring_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)
    lines = _read_lines(tmp_path / "pair.txt")

    scores, _ = calc_convergence_pair_scores(large_ring_camset)
    n_views = len(large_ring_camset)

    for i in range(n_views):
        _, got_ids, _ = _parse_pair_row(lines, i)
        assert got_ids == _top_k_ranking(scores, i, n_views, _APDE_MVS_MAX_SRC_VIEWS)


def test_max_src_views_is_configurable(large_ring_camset, tmp_path):
    """A caller building against a recompiled MAX_IMAGES must be able to
    move the cap, not just live with the APD-MVS/APDe-MVS default."""
    camset_to_apde(large_ring_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64, max_src_views=5)
    lines = _read_lines(tmp_path / "pair.txt")

    for i in range(len(large_ring_camset)):
        n_neighbours, _, _ = _parse_pair_row(lines, i)
        assert n_neighbours == 5


def test_small_rig_is_unaffected_by_the_cap(apde_camset, tmp_path):
    """The default cap (31) must not change output for any rig smaller than
    it -- e.g. the 3-camera fixture every other test in this file uses."""
    camset_to_apde(apde_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)
    lines = _read_lines(tmp_path / "pair.txt")

    n_views = len(apde_camset)
    for i in range(n_views):
        n_neighbours, _, _ = _parse_pair_row(lines, i)
        assert n_neighbours == n_views - 1  # every other view, uncapped


def test_write_to_txt_max_pair_candidates_caps_and_keeps_top_scores(large_ring_camset, tmp_path):
    """The cap is implemented in CameraSet.write_to_txt itself (camset_to_apde
    just supplies the APD-MVS/APDe-MVS-specific default), so it must work
    when called directly too."""
    from pyCamSet.reconstruction.acmmp_utils import ReconParams, calc_convergence_pair_scores

    scores, _ = calc_convergence_pair_scores(large_ring_camset)
    cams_dir = tmp_path / "cams"
    cams_dir.mkdir(parents=True)
    r = ReconParams(mindist=0.2, maxdist=1.0, steps=64)
    large_ring_camset.write_to_txt(cams_dir, r, pair_scores=scores, max_pair_candidates=10)

    lines = _read_lines(tmp_path / "pair.txt")
    n_views = len(large_ring_camset)
    for i in range(n_views):
        n_neighbours, got_ids, _ = _parse_pair_row(lines, i)
        assert n_neighbours == 10
        assert got_ids == _top_k_ranking(scores, i, n_views, 10)


def test_truncation_logs_a_warning_naming_how_many_were_dropped(large_ring_camset, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="pyCamSet.utils.saving"):
        camset_to_apde(large_ring_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)

    assert "40" in caplog.text  # camera count
    assert "31" in caplog.text  # the cap
    assert "8" in caplog.text  # 39 other views - 31 kept = 8 dropped


def test_no_truncation_warning_under_the_cap(apde_camset, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="pyCamSet.utils.saving"):
        camset_to_apde(apde_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64)

    assert "cap" not in caplog.text.lower()


# --------------------------------------------------------------------------
# max_src_views / max_pair_candidates validation (DEFECT 4)
#
# A negative value is not itself an error under Python's slice semantics --
# `row[:-1]` silently drops only the last element instead of capping the
# list -- so a caller's own bug (e.g. an off-by-one computing a cap from a
# rebuilt tool's MAX_IMAGES) would reintroduce the near-unbounded, crash-
# triggering output this whole cap exists to prevent, without pyCamSet ever
# raising. Validated eagerly, before anything is written, so a bad value
# never leaves a half-exported directory behind either.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [-1, -31, 1.5, "31", True])
def test_camset_to_apde_rejects_invalid_max_src_views(apde_camset, tmp_path, bad_value):
    with pytest.raises(ValueError):
        camset_to_apde(apde_camset, tmp_path, depth_min=0.2, depth_max=1.0, depth_num=64, max_src_views=bad_value)
    # rejected before anything was written -- no half-exported directory left behind
    assert not (tmp_path / "cams").exists()
    assert not (tmp_path / "pair.txt").exists()


@pytest.mark.parametrize("bad_value", [-1, -5, 1.5, "5", True])
def test_write_to_txt_rejects_invalid_max_pair_candidates(apde_camset, tmp_path, bad_value):
    from pyCamSet.reconstruction.acmmp_utils import ReconParams, calc_convergence_pair_scores

    scores, _ = calc_convergence_pair_scores(apde_camset)
    cams_dir = tmp_path / "cams"
    cams_dir.mkdir(parents=True)
    r = ReconParams(mindist=0.2, maxdist=1.0, steps=64)
    with pytest.raises(ValueError):
        apde_camset.write_to_txt(cams_dir, r, pair_scores=scores, max_pair_candidates=bad_value)
    # rejected before any *_cam.txt was written -- no partial export
    assert list(cams_dir.glob("*_cam.txt")) == []


def test_max_pair_candidates_zero_is_allowed_and_empties_every_row(apde_camset, tmp_path):
    from pyCamSet.reconstruction.acmmp_utils import ReconParams, calc_convergence_pair_scores

    scores, _ = calc_convergence_pair_scores(apde_camset)
    cams_dir = tmp_path / "cams"
    cams_dir.mkdir(parents=True)
    r = ReconParams(mindist=0.2, maxdist=1.0, steps=64)
    apde_camset.write_to_txt(cams_dir, r, pair_scores=scores, max_pair_candidates=0)

    lines = _read_lines(tmp_path / "pair.txt")
    for i in range(len(apde_camset)):
        n_neighbours, ids, row_scores = _parse_pair_row(lines, i)
        assert n_neighbours == 0
        assert ids == []
        assert row_scores == []
