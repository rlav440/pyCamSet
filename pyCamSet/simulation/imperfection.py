"""
How a printed net sits on a real solid.

A person cutting out a net's faces and pasting them onto a 3D-printed core
never places them perfectly.  Each face's paper ends up a little turned,
a little shifted, and -- from the printer and the paper itself -- a little
off scale.  None of that changes what the target *says* its points are, so a
calibration against the nominal ``point_data`` carries the error as residual;
the true points the images were drawn from are what :func:`paste_faces`
returns.

Everything here works in a face's own frame: ``(x, y)`` in metres on the
face, as the target's ``faceData.face_local_coords`` holds them, which every
cube target in pyCamSet shares -- the face's top-left corner at the origin, x
to the right, y down, the face spanning ``[0, length]`` both ways.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PastingErrors:
    """
    How badly each face's paper was placed.

    :param rotation_deg: each face's paper turned about the face's centre,
        in degrees, ``(n_faces,)``.
    :param offset: each face's paper shifted within its face, in metres,
        ``(n_faces, 2)`` in the face's own ``(x, y)``.
    :param scale: each face's paper scaled about the face's centre,
        ``(n_faces, 2)`` -- 1.0 is exact; ``x`` and ``y`` separately, since
        paper stretches more across the grain than along it.
    """

    rotation_deg: np.ndarray
    offset: np.ndarray
    scale: np.ndarray

    @classmethod
    def none(cls, n_faces: int = 6) -> "PastingErrors":
        """A perfectly made target."""
        return cls(np.zeros(n_faces), np.zeros((n_faces, 2)), np.ones((n_faces, 2)))

    @classmethod
    def by_hand(cls, n_faces: int = 6, rotation_deg: float = 0.5, offset: float = 0.3e-3,
                print_scale: float = 0.002, stretch: float = 0.001,
                seed: int = 2) -> "PastingErrors":
        """
        Errors typical of careful pasting by hand.

        Each face's rotation and offset are drawn from a normal distribution
        of the given standard deviation.  ``print_scale`` is one error the
        whole net shares -- a printer's scale is off by up to a few tenths of
        a percent -- drawn uniformly within it; ``stretch`` is each face's own
        paper stretch on top, per axis, drawn normally.

        :param n_faces: how many faces.
        :param rotation_deg: standard deviation of each face's turn, degrees.
        :param offset: standard deviation of each face's shift, metres, per
            axis.
        :param print_scale: the largest printer scale error, as a fraction.
        :param stretch: standard deviation of each face's stretch, as a
            fraction, per axis.
        :param seed: the random seed.
        """
        rng = np.random.default_rng(seed)
        shared = 1.0 + rng.uniform(-print_scale, print_scale)
        return cls(
            rotation_deg=rng.normal(0.0, rotation_deg, n_faces),
            offset=rng.normal(0.0, offset, (n_faces, 2)),
            scale=shared * (1.0 + rng.normal(0.0, stretch, (n_faces, 2))),
        )

    def to_dict(self) -> dict:
        return {"rotation_deg": self.rotation_deg.tolist(), "offset_m": self.offset.tolist(),
                "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "PastingErrors":
        return cls(np.asarray(data["rotation_deg"], float),
                   np.asarray(data["offset_m"], float), np.asarray(data["scale"], float))

    def paper_to_face(self, face_index: int, length: float) -> np.ndarray:
        """
        The 3x3 affine taking a point printed on face ``face_index``'s paper
        (in the net's face frame) to where it ends up on the solid's face.

        :param face_index: which face.
        :param length: the face's edge, in metres -- its centre is the point
            the paper turns and scales about.
        """
        centre = np.array([length / 2.0, length / 2.0])
        theta = np.radians(self.rotation_deg[face_index])
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        linear = rotation @ np.diag(self.scale[face_index])
        affine = np.eye(3)
        affine[:2, :2] = linear
        affine[:2, 2] = centre - linear @ centre + self.offset[face_index]
        return affine


def face_to_world(target, face_index: int, local_xy) -> np.ndarray:
    """
    Points on a face, in the face's frame, placed on the solid as the target
    places its own points.

    :param target: a target built on
        ``FaceToShape`` (``pyCamSet.calibration_targets.core.shape_by_faces``),
        with ``faceData``.
    :param face_index: which face.
    :param local_xy: an ``(N, 2)`` array in metres.
    :return: an ``(N, 3)`` array in the target's own frame.
    """
    shape = target.faceData
    transform = shape.ur_ftform[face_index]
    local = np.c_[np.asarray(local_xy, dtype=float), np.zeros(len(local_xy))]
    # FaceToShape places a face as sf * T(local / sf).
    return local @ transform[:3, :3].T + shape.sf * transform[:3, 3]


def paste_faces(target, errors: PastingErrors) -> np.ndarray:
    """
    Where the target's points really are, on a solid its faces were pasted
    onto with ``errors``.

    :param target: the target, nominal.
    :param errors: how each face was placed.
    :return: an array shaped like ``target.point_data``.
    """
    shape = target.faceData
    true_points = np.empty_like(np.asarray(target.point_data, dtype=float))
    flat = true_points.reshape(len(shape.ur_ftform), -1, 3)
    length = float(target.length)
    for face, local in enumerate(shape.ur_flocal):
        moved = np.c_[local[:, :2], np.ones(len(local))] @ errors.paper_to_face(face, length).T
        flat[face] = face_to_world(target, face, moved[:, :2])
    return true_points
