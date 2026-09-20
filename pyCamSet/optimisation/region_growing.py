"""
An initial rig estimate grown from the camera links the data agrees about.

A rigid rig means every image shared by two cameras reports the same transform
between them.  The candidates are therefore measurable against each other, and
how far they disagree is that link's reliability -- measured rather than
guessed from how a single view looks.  The rig is grown outwards from the most
agreed link, so error accumulates along the most trustworthy route available
rather than along whichever one a shortest path search happened to return.

This matters where a route is long.  A rig whose cameras all see the target
together is one hop from anywhere and any route will do; a rig that only
overlaps between neighbours composes a transform per link, and which link
estimate is used at each step decides the result.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

#: Candidates compared per link.  Agreement is measured between every pair of
#: them, so the work is quadratic, and a link is characterised long before
#: every shared image has been looked at.
MAX_LINK_SAMPLES = 24


def invert_rigid(transforms: np.ndarray) -> np.ndarray:
    """
    The inverse of a stack of rigid transforms, by transpose rather than solve.

    :param transforms: (..., 4, 4) rigid transforms
    :return: their inverses, in the same shape
    """
    rotation = np.swapaxes(transforms[..., :3, :3], -1, -2)
    translation = transforms[..., :3, 3]

    inverted = np.zeros_like(transforms)
    inverted[..., :3, :3] = rotation
    inverted[..., :3, 3] = -np.einsum("...ij,...j->...i", rotation, translation)
    inverted[..., 3, 3] = 1.0
    return inverted


def disagreement(transforms: np.ndarray, length_scale: float) -> np.ndarray:
    """
    How far apart a set of rigid transforms are, pairwise, in world units.

    An angle is counted as the distance it moves a point *length_scale* from
    the origin, which puts rotation and translation on one scale without a
    weighting to choose.

    :param transforms: (n, 4, 4) rigid transforms
    :param length_scale: the radius of the object being placed
    :return: an (n, n) distance
    """
    delta = invert_rigid(transforms)[:, None] @ transforms[None, :]
    trace = np.trace(delta[..., :3, :3], axis1=-2, axis2=-1)
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    return np.hypot(np.linalg.norm(delta[..., :3, 3], axis=-1),
                    length_scale * angle)


def consensus(transforms: np.ndarray, length_scale: float
              ) -> tuple[np.ndarray, float]:
    """
    The transform the candidates agree on, and how much they disagree.

    The medoid is returned rather than an average of the candidates that agree
    with it.  A planar target resolves to two poses that both explain the
    image, and a mean over candidates that straddle the two is a pose that
    matches neither; the medoid is always one of the measurements.

    :param transforms: (n, 4, 4) candidate transforms
    :param length_scale: the radius of the object being placed
    :return: the agreed transform, and the median disagreement about it, which
        is NaN for a single candidate: it is the only thing said, so nothing
        disagrees with it and nothing confirms it either
    """
    if len(transforms) == 1:
        return transforms[0], np.nan

    distances = disagreement(transforms, length_scale)
    medoid = int(np.argmin(np.sum(distances, axis=1)))
    return transforms[medoid], float(np.median(distances[medoid]))


def camera_links(target_in_camera: np.ndarray, observed: np.ndarray,
                 length_scale: float, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    The transform between each pair of cameras, and how well it is agreed.

    A pair sharing a single image gets a link with nothing to check it
    against.  That is worth less than any link the images agree on, but it is
    worth much more than no link at all, so it is scored behind every measured
    link rather than discarded as unmeasurable.

    :param target_in_camera: (c, p, 4, 4) target pose as each camera saw it
    :param observed: (c, p) whether that view produced a pose
    :param length_scale: the radius of the target
    :param rng: samples the shared images when there are many
    :return: (c, c, 4, 4) camera to camera transforms, and their (c, c)
        disagreement, infinite where there is no link
    """
    n_cams = len(target_in_camera)
    links = np.zeros((n_cams, n_cams, 4, 4))
    spread = np.full((n_cams, n_cams), np.inf)

    for first in range(n_cams):
        for second in range(first + 1, n_cams):
            shared = np.nonzero(observed[first] & observed[second])[0]
            if shared.size == 0:
                continue
            if shared.size > MAX_LINK_SAMPLES:
                shared = rng.choice(shared, MAX_LINK_SAMPLES, replace=False)

            candidates = (target_in_camera[second, shared]
                          @ invert_rigid(target_in_camera[first, shared]))
            agreed, scatter = consensus(candidates, length_scale)

            links[first, second] = agreed
            links[second, first] = invert_rigid(agreed[None])[0]
            spread[first, second] = spread[second, first] = scatter

    unchecked = np.isnan(spread)
    measured = np.isfinite(spread)
    spread[unchecked] = np.max(spread[measured]) if np.any(measured) else 1.0
    return links, spread


def camera_groups(spread: np.ndarray) -> np.ndarray:
    """
    Which group of mutually connected cameras each camera falls in.

    :param spread: (c, c) link disagreement, infinite where there is no link
    :return: a (c,) group label per camera
    """
    return connected_components(csr_matrix(np.isfinite(spread)),
                                directed=False)[1]


def read_out(items: list[str]) -> str:
    """
    Join items the way they would be read aloud.

    :param items: the already formatted items
    :return: "a", "a and b", or "a, b and c"
    """
    if len(items) < 3:
        return " and ".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def describe_split(groups: np.ndarray, observed: np.ndarray,
                   cam_names: list[str] | None) -> str:
    """
    Say which cameras never met, in the terms a person can act on.

    :param groups: a group label per camera
    :param observed: (c, p) whether each camera placed the target in each image
    :param cam_names: the camera names, defaulting to their indices
    :return: the message for the refusal
    """
    names = (list(cam_names) if cam_names
             else [str(index) for index in range(len(groups))])
    labels = np.unique(groups)
    listed = read_out([
        "{" + ", ".join(names[i] for i in np.nonzero(groups == label)[0]) + "}"
        for label in labels])

    message = (
        f"the cameras fall into {len(labels)} groups that never saw the target "
        f"at the same time -- {listed} -- so there is no transform between "
        f"them to solve for. Capture images in which a camera from each group "
        f"sees the target together, or calibrate the groups separately")

    blind = [names[i] for i in np.nonzero(~np.any(observed, axis=1))[0]]
    if blind:
        message += (f". {read_out(blind)} placed the target in no image at "
                    f"all, which is usually the wrong folder or the wrong "
                    f"target rather than the images")
    return message


def grow_cameras(links: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """
    Place the cameras, joining each one by the link it is most agreed with.

    :param links: (c, c, 4, 4) camera to camera transforms
    :param spread: (c, c) their disagreement, infinite where there is no link
    :return: (c, 4, 4) world to camera, NaN for a camera outside the group
        that was grown
    """
    n_cams = len(spread)
    linked = np.isfinite(spread)

    extrinsics = np.full((n_cams, 4, 4), np.nan)
    placed = np.zeros(n_cams, dtype=bool)
    root = int(np.argmax(np.sum(linked, axis=1)))
    extrinsics[root], placed[root] = np.eye(4), True

    while not np.all(placed):
        reachable = spread[np.ix_(placed, ~placed)]
        if not np.any(np.isfinite(reachable)):
            break
        inside, outside = np.nonzero(placed)[0], np.nonzero(~placed)[0]
        best = int(np.argmin(np.where(np.isfinite(reachable), reachable, np.inf)))
        joined_from = inside[best // len(outside)]
        joining = outside[best % len(outside)]

        extrinsics[joining] = links[joined_from, joining] @ extrinsics[joined_from]
        placed[joining] = True
    return extrinsics


def target_poses(target_in_camera: np.ndarray, observed: np.ndarray,
                 extrinsics: np.ndarray, length_scale: float) -> np.ndarray:
    """
    Place each image, as the cameras that saw it agree on where it was.

    :param target_in_camera: (c, p, 4, 4) target pose as each camera saw it
    :param observed: (c, p) whether that view produced a pose
    :param extrinsics: (c, 4, 4) world to camera
    :param length_scale: the radius of the target
    :return: (p, 4, 4) target to world, NaN for an image no placed camera saw
    """
    placed = np.isfinite(extrinsics[:, 0, 0])
    world_from_camera = invert_rigid(
        np.where(np.isnan(extrinsics), np.eye(4), extrinsics))

    poses = np.full((target_in_camera.shape[1], 4, 4), np.nan)
    for image in range(target_in_camera.shape[1]):
        views = np.nonzero(observed[:, image] & placed)[0]
        if views.size == 0:
            continue
        poses[image] = consensus(
            world_from_camera[views] @ target_in_camera[views, image],
            length_scale)[0]
    return poses


def anchored_on(extrinsics: np.ndarray, poses: np.ndarray, reference: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Move the world frame onto one image, leaving every observation unchanged.

    Growing leaves the world frame on the root camera, but the bundle
    adjustment holds one target pose fixed, so an estimate anchored anywhere
    else is held against a gauge it was never given.

    :param extrinsics: (c, 4, 4) world to camera
    :param poses: (p, 4, 4) target to world
    :param reference: the image to anchor on, if it was placed
    :return: the re-anchored extrinsics and poses
    """
    solved = np.isfinite(poses[:, 0, 0])
    if not np.any(solved):
        return extrinsics, poses
    if not (0 <= reference < len(poses) and solved[reference]):
        reference = int(np.argmax(solved))

    anchor = poses[reference].copy()
    poses = poses.copy()
    poses[solved] = invert_rigid(anchor) @ poses[solved]

    extrinsics = extrinsics.copy()
    placed = np.isfinite(extrinsics[:, 0, 0])
    extrinsics[placed] = extrinsics[placed] @ anchor
    return extrinsics, poses


def estimate_rig(target_in_camera: np.ndarray, length_scale: float,
                 reference_image: int = 0, seed: int = 0,
                 cam_names: list[str] | None = None,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Place every camera and every image from the per view target poses.

    :param target_in_camera: (c, p, 4, 4) target pose as each camera saw it,
        NaN where that camera could not place the target in that image
    :param length_scale: the radius of the target
    :param reference_image: the image to anchor the world frame on
    :param seed: samples the shared images of a heavily overlapped link
    :param cam_names: camera names, for naming the cameras in a refusal
    :raises ValueError: when shared views of the target do not connect every
        camera, which leaves the rig with no transform to solve for
    :return: (c, 4, 4) world to camera, and (p, 4, 4) target to world, the
        latter NaN for any image no camera placed the target in
    """
    observed = np.isfinite(target_in_camera[:, :, 0, 0])
    rng = np.random.default_rng(seed)

    links, spread = camera_links(target_in_camera, observed, length_scale, rng)

    groups = camera_groups(spread)
    if np.max(groups) > 0:
        raise ValueError(describe_split(groups, observed, cam_names))

    extrinsics = grow_cameras(links, spread)
    poses = target_poses(target_in_camera, observed, extrinsics, length_scale)
    return anchored_on(extrinsics, poses, reference_image)
