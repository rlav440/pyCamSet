"""The initial rig estimate, grown from the links the data agrees about.

These are all synthetic and exact: a rig is invented, the target poses each
camera would have measured are composed directly, and the estimate has to give
the rig back. That keeps the failures readable -- a wrong answer here is a
wrong transform, not a detector that had a bad day.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyCamSet.optimisation.region_growing import (
    anchored_on,
    camera_links,
    consensus,
    disagreement,
    estimate_rig,
    grow_cameras,
    invert_rigid,
    camera_groups,
    read_out,
    target_poses,
)

RADIUS = 40.0


def rigid(rotvec, translation) -> np.ndarray:
    tform = np.eye(4)
    tform[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    tform[:3, 3] = translation
    return tform


@pytest.fixture
def rig():
    """Four cameras and six target poses, as exact transforms."""
    extrinsics = np.array([
        rigid([0.0, 0.1 * index, 0.0], [60.0 * index, 0.0, 0.0])
        for index in range(4)])
    poses = np.array([
        rigid([0.1 * index, -0.2, 0.05 * index], [5.0 * index, 3.0, 400.0])
        for index in range(6)])
    target_in_camera = extrinsics[:, None] @ poses[None, :]
    return extrinsics, poses, target_in_camera


def relative_to_first(transforms):
    return np.array([t @ np.linalg.inv(transforms[0]) for t in transforms])


# --------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------


def test_invert_rigid_is_an_exact_inverse():
    transforms = np.array([rigid([0.3, -0.2, 0.1], [4.0, -5.0, 6.0]),
                           rigid([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])])

    assert np.allclose(invert_rigid(transforms) @ transforms, np.eye(4))


def test_identical_transforms_do_not_disagree():
    repeated = np.repeat(rigid([0.2, 0.0, 0.0], [1.0, 2.0, 3.0])[None], 3, axis=0)

    assert np.allclose(disagreement(repeated, RADIUS), 0.0)


def test_a_rotation_is_counted_as_the_distance_it_moves_the_rim():
    angle = 0.02
    pair = np.array([np.eye(4), rigid([0.0, 0.0, angle], [0.0, 0.0, 0.0])])

    assert disagreement(pair, RADIUS)[0, 1] == pytest.approx(RADIUS * angle)


def test_consensus_ignores_a_single_disagreeing_candidate():
    agreed = rigid([0.1, 0.0, 0.0], [1.0, 0.0, 0.0])
    candidates = np.array([agreed, agreed, agreed,
                           rigid([2.0, 1.0, 0.0], [500.0, 0.0, 0.0])])

    centre, scatter = consensus(candidates, RADIUS)

    assert np.allclose(centre, agreed)
    assert scatter == pytest.approx(0.0)


def test_consensus_returns_a_measurement_not_an_average():
    """A planar target resolves to two poses; a mean of the two is neither."""
    first = rigid([0.4, 0.0, 0.0], [0.0, 0.0, 0.0])
    second = rigid([-0.4, 0.0, 0.0], [0.0, 0.0, 0.0])

    centre, _ = consensus(np.array([first, second, second]), RADIUS)

    assert np.allclose(centre, first) or np.allclose(centre, second)


def test_a_single_candidate_has_nothing_to_disagree_with():
    only = rigid([0.1, 0.0, 0.0], [1.0, 0.0, 0.0])

    centre, scatter = consensus(only[None], RADIUS)

    assert np.allclose(centre, only)
    assert np.isnan(scatter)


# --------------------------------------------------------------------------
# Links and growth
# --------------------------------------------------------------------------


def test_a_link_is_the_transform_between_the_cameras(rig):
    extrinsics, _, target_in_camera = rig
    observed = np.ones(target_in_camera.shape[:2], dtype=bool)

    links, spread = camera_links(target_in_camera, observed, RADIUS,
                                 np.random.default_rng(0))

    expected = extrinsics[2] @ np.linalg.inv(extrinsics[1])
    assert np.allclose(links[1, 2], expected)
    # arccos loses half the available digits next to the identity, so an exact
    # agreement measures as micrometres rather than as zero
    assert np.allclose(spread[np.isfinite(spread)], 0.0, atol=1e-5)


def test_a_link_from_one_shared_image_is_used_but_ranked_last(rig):
    """One shared image is unverifiable, not unusable: a sparse rig has
    nothing else, and scoring it as no link at all disconnects the rig."""
    _, _, target_in_camera = rig
    observed = np.ones(target_in_camera.shape[:2], dtype=bool)
    observed[3, 1:] = False          # camera 3 shares only image 0
    target_in_camera = np.where(observed[:, :, None, None],
                                target_in_camera, np.nan)

    _, spread = camera_links(target_in_camera, observed, RADIUS,
                             np.random.default_rng(0))

    assert np.isfinite(spread[0, 3])
    assert spread[0, 3] >= np.max(spread[:3, :3][np.isfinite(spread[:3, :3])])


def test_a_camera_nothing_connects_to_is_its_own_group(rig):
    _, _, target_in_camera = rig
    observed = np.ones(target_in_camera.shape[:2], dtype=bool)
    observed[3] = False
    target_in_camera = np.where(observed[:, :, None, None],
                                target_in_camera, np.nan)

    _, spread = camera_links(target_in_camera, observed, RADIUS,
                             np.random.default_rng(0))
    groups = camera_groups(spread)

    assert groups[3] not in groups[:3]
    assert len(set(groups[:3])) == 1


def test_growth_recovers_the_relative_camera_poses(rig):
    extrinsics, _, target_in_camera = rig
    observed = np.ones(target_in_camera.shape[:2], dtype=bool)

    links, spread = camera_links(target_in_camera, observed, RADIUS,
                                 np.random.default_rng(0))
    grown = grow_cameras(links, spread)

    assert np.allclose(relative_to_first(grown), relative_to_first(extrinsics))


def test_an_image_no_placed_camera_saw_has_no_pose(rig):
    extrinsics, _, target_in_camera = rig
    observed = np.ones(target_in_camera.shape[:2], dtype=bool)
    observed[:, 4] = False
    target_in_camera = np.where(observed[:, :, None, None],
                                target_in_camera, np.nan)

    poses = target_poses(target_in_camera, observed, extrinsics, RADIUS)

    assert np.all(np.isnan(poses[4]))
    assert np.all(np.isfinite(np.delete(poses, 4, axis=0)))


# --------------------------------------------------------------------------
# Gauge
# --------------------------------------------------------------------------


def test_anchoring_puts_the_reference_image_at_the_origin(rig):
    extrinsics, poses, _ = rig

    _, anchored = anchored_on(extrinsics, poses, 3)

    assert np.allclose(anchored[3], np.eye(4))


def test_anchoring_leaves_every_observation_unchanged(rig):
    extrinsics, poses, target_in_camera = rig

    moved_ext, moved_poses = anchored_on(extrinsics, poses, 3)

    assert np.allclose(moved_ext[:, None] @ moved_poses[None, :],
                       target_in_camera)


def test_anchoring_falls_back_when_the_reference_was_not_placed(rig):
    extrinsics, poses, _ = rig
    poses = poses.copy()
    poses[0] = np.nan

    _, anchored = anchored_on(extrinsics, poses, 0)

    assert np.allclose(anchored[1], np.eye(4))


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_the_rig_is_recovered_from_its_own_observations(rig):
    extrinsics, poses, target_in_camera = rig

    grown, grown_poses = estimate_rig(target_in_camera, RADIUS,
                                      reference_image=0)

    assert np.allclose(relative_to_first(grown), relative_to_first(extrinsics))
    assert np.allclose(grown[:, None] @ grown_poses[None, :], target_in_camera)


def test_a_chained_rig_is_recovered_through_its_neighbours(rig):
    """No camera sees what the far end sees, so every transform is composed."""
    extrinsics, _, target_in_camera = rig
    observed = np.zeros(target_in_camera.shape[:2], dtype=bool)
    for image in range(target_in_camera.shape[1]):
        pair = image % (len(extrinsics) - 1)
        observed[pair, image] = observed[pair + 1, image] = True
    chained = np.where(observed[:, :, None, None], target_in_camera, np.nan)

    grown, _ = estimate_rig(chained, RADIUS, reference_image=0)

    assert np.allclose(relative_to_first(grown), relative_to_first(extrinsics))


def test_a_camera_that_saw_nothing_is_refused_by_name(rig):
    _, _, target_in_camera = rig
    target_in_camera = target_in_camera.copy()
    target_in_camera[3] = np.nan

    with pytest.raises(ValueError, match="lonely"):
        estimate_rig(target_in_camera, RADIUS,
                     cam_names=["a", "b", "c", "lonely"])


# --------------------------------------------------------------------------
# Rigs that shared views do not join up
# --------------------------------------------------------------------------


def split_rig(group_sizes=(3, 5), n_images=12):
    """Groups of cameras that never see the target at the same time."""
    n_cams = sum(group_sizes)
    extrinsics = np.array([rigid([0.0, 0.1 * i, 0.0], [50.0 * i, 0.0, 0.0])
                           for i in range(n_cams)])
    poses = np.array([rigid([0.05 * i, -0.2, 0.02 * i], [4.0 * i, 3.0, 400.0])
                      for i in range(n_images)])

    observed = np.zeros((n_cams, n_images), dtype=bool)
    per_group = n_images // len(group_sizes)
    first = 0
    for index, size in enumerate(group_sizes):
        observed[first:first + size,
                 index * per_group:(index + 1) * per_group] = True
        first += size

    target_in_camera = np.where(
        observed[:, :, None, None], extrinsics[:, None] @ poses[None, :], np.nan)
    return extrinsics, target_in_camera, observed


def test_the_groups_a_split_rig_falls_into_are_found():
    _, target_in_camera, observed = split_rig()

    _, spread = camera_links(target_in_camera, observed, RADIUS,
                             np.random.default_rng(0))
    groups = camera_groups(spread)

    assert len(set(groups[:3])) == 1
    assert len(set(groups[3:])) == 1
    assert groups[0] != groups[3]


def test_a_split_rig_is_refused():
    """There is no transform between groups that never saw the target
    together, so placing one of them and leaving the rest at the origin would
    answer a question the images cannot answer."""
    _, target_in_camera, _ = split_rig(group_sizes=(3, 5))

    with pytest.raises(ValueError, match="never saw the target at the same time"):
        estimate_rig(target_in_camera, RADIUS, reference_image=0)


def test_the_refusal_names_both_groups():
    _, target_in_camera, _ = split_rig(group_sizes=(3, 5))

    with pytest.raises(ValueError) as refusal:
        estimate_rig(target_in_camera, RADIUS,
                     cam_names=[f"c{i}" for i in range(8)])

    message = str(refusal.value)
    assert "{c0, c1, c2}" in message
    assert "{c3, c4, c5, c6, c7}" in message
    assert "calibrate the groups separately" in message


def test_a_connected_rig_is_not_refused(rig):
    _, _, target_in_camera = rig

    estimate_rig(target_in_camera, RADIUS, reference_image=0)


@pytest.mark.parametrize("group_sizes", [(3, 5), (2, 2, 2), (1, 1, 1, 1)])
def test_a_rig_in_any_number_of_pieces_is_refused(group_sizes):
    _, target_in_camera, _ = split_rig(group_sizes)
    names = [f"c{index}" for index in range(sum(group_sizes))]

    with pytest.raises(ValueError) as refusal:
        estimate_rig(target_in_camera, RADIUS, cam_names=names)

    message = str(refusal.value)
    assert f"fall into {len(group_sizes)} groups" in message
    assert all(name in message for name in names)


def test_every_group_is_named_when_there_are_more_than_two():
    _, target_in_camera, _ = split_rig((2, 2, 2))

    with pytest.raises(ValueError) as refusal:
        estimate_rig(target_in_camera, RADIUS,
                     cam_names=[f"c{i}" for i in range(6)])

    assert "{c0, c1}, {c2, c3} and {c4, c5}" in str(refusal.value)


@pytest.mark.parametrize("items, expected", [
    (["a"], "a"),
    (["a", "b"], "a and b"),
    (["a", "b", "c"], "a, b and c"),
    (["a", "b", "c", "d"], "a, b, c and d"),
])
def test_items_are_joined_the_way_they_would_be_read_aloud(items, expected):
    assert read_out(items) == expected
