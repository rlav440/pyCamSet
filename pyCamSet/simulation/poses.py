"""
Where the target is held for each shot.
"""

from __future__ import annotations

import cv2
import numpy as np


def held_target_poses(n_shots: int = 30, translation: float = 0.02,
                      seed: int = 1) -> np.ndarray:
    """
    Target-to-world transforms for a target turned in the hand in front of
    the rig.

    A cube is shown to the cameras in every orientation over a capture, so
    the rotations are drawn uniformly over all of them (a random unit
    quaternion each), and the cube is held near the origin, within
    ``translation`` metres of it on each axis.  The cube's own origin is its
    centre, so this turns it about itself.

    :param n_shots: how many shots.
    :param translation: how far from the origin the cube wanders, in metres.
    :param seed: the random seed.
    :return: an ``(n_shots, 4, 4)`` array.
    """
    rng = np.random.default_rng(seed)
    poses = np.repeat(np.eye(4)[None], n_shots, axis=0)
    for shot in range(n_shots):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
        axis = q[1:] / max(np.linalg.norm(q[1:]), 1e-12)
        poses[shot, :3, :3] = cv2.Rodrigues(axis * angle)[0]
        poses[shot, :3, 3] = rng.uniform(-translation, translation, 3)
    return poses
