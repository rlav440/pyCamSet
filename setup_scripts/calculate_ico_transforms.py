"""Generates the face and net transforms for the icosahedral targets.

A design-time tool, not library code.  Its output is the ``ICO_TFORMS`` and
``ICO_NET_FORMS`` constants checked in at
``pyCamSet/calibration_targets/polyhedra.py``, so the package already carries
the result and nothing imports this at runtime.  Run it directly to regenerate
them, alongside calculate_shape_transforms.py and make_target_net.py.

Unlike the cube's generator this does not go through
:func:`~pyCamSet.calibration_targets.core.shape_by_faces.make_tforms`.  That
helper sizes its polyhedron from the base face's y extent, which for a square
is the edge length but for an equilateral triangle is the *height* -- and it
then hands that number to ``pv.Icosahedron`` as a *radius*, where the edge is
``radius / sin(2*pi/5)``.  The two disagree by about 9%, and
``n_estimate_rigid_transform`` returns a rigid transform, so it cannot absorb
the difference: it would quietly return a best fit with residual rather than
the exact placement we want.  The icosahedron is built analytically here
instead, which also fixes its orientation.

That orientation matters.  ``pv.Icosahedron`` is turned so that its twenty
face centroids sit at seven distinct heights, which gives no useful way to
group the faces.  Built with a vertex at each pole the faces fall into four
rings of five -- top cap, upper band, lower band, bottom cap -- which is the
order the faces are numbered in, and which makes the net a strip with a row of
caps hung off each side.

Unlike make_target_net.py, whose docstring warns that a fresh run permutes two
of the cube's faces, this generator is deterministic and its output is checked
against the checked-in constants by
``tests/test_polyhedra.py::test_the_checked_in_constants_are_what_the_generator_makes``.
"""

import contextlib
import io
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from make_target_net import make_net_tforms  # noqa: E402

from pyCamSet.optimisation.compiled_helpers import (  # noqa: E402
    n_estimate_rigid_transform,
)
from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform  # noqa: E402

#: Half the height of a unit equilateral triangle's bounding box.
TRIANGLE_HEIGHT = np.sqrt(3) / 2

#: The base face, edge length one, wound anticlockwise seen from +z.
BASE_TRIANGLE_3D = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, TRIANGLE_HEIGHT, 0.0]])

#: The same triangle in the plane, for the net.
BASE_TRIANGLE_2D = BASE_TRIANGLE_3D[:, :2].copy()


def icosahedron(edge_length: float = 1.0):
    """
    Build an icosahedron with a vertex at each pole.

    :param edge_length: the edge length every face is built to
    :return: its twelve vertices and its twenty faces, wound so that each
        face's normal points away from the centre
    """
    radius = edge_length * np.sin(2 * np.pi / 5)
    cos_a, sin_a = 1 / np.sqrt(5), 2 / np.sqrt(5)

    vertices = [np.array([0.0, 0.0, 1.0]) * radius]
    for ring, tilt in ((0.0, cos_a), (np.pi / 5, -cos_a)):
        for i in range(5):
            angle = 2 * np.pi * i / 5 + ring
            vertices.append(radius * np.array(
                [sin_a * np.cos(angle), sin_a * np.sin(angle), tilt]))
    vertices.append(np.array([0.0, 0.0, -1.0]) * radius)

    def upper(i):
        return 1 + i % 5

    def lower(i):
        return 6 + i % 5

    faces = []
    for i in range(5):
        faces.append([0, upper(i), upper(i + 1)])
    for i in range(5):
        faces.append([upper(i), lower(i), upper(i + 1)])
    for i in range(5):
        faces.append([lower(i), lower(i + 1), upper(i + 1)])
    for i in range(5):
        faces.append([11, lower(i + 1), lower(i)])
    return np.array(vertices), np.array(faces)


def face_transforms(vertices, faces):
    """
    Return the transform placing the base triangle on each face.

    :param vertices: the solid's vertices
    :param faces: the vertex indices of each face
    :return: one ``(rotation vector, translation)`` pair per face, and the
        largest distance any placed corner missed its vertex by
    """
    transforms, residuals = [], []
    for face in faces:
        rotation, translation = n_estimate_rigid_transform(
            BASE_TRIANGLE_3D, vertices[face])
        rvec, _ = cv2.Rodrigues(rotation)
        transforms.append((rvec.squeeze(), translation.squeeze()))
        placed = h_tform(BASE_TRIANGLE_3D, make_4x4h_tform(rvec, translation))
        residuals.append(np.abs(placed - vertices[face]).max())
    return transforms, max(residuals)


def net_transforms(faces):
    """
    Return the transform placing each face in the printable net.

    The net is the strip the four rings make: the ten band faces zigzag
    across, with the top caps hung above their band face and the bottom caps
    below theirs.  Unfolding is rigid across a shared edge, so the net folds
    back to the solid the face transforms describe.

    :param faces: the vertex indices of each face
    :return: one 3x3 affine per face, in the row/column convention
        :meth:`FaceToShape.draw_net` uses
    """
    connectivity = []
    for face in faces:
        connectivity.extend([3, *face])

    # Ring k occupies faces 5k to 5k+4; see icosahedron().
    def top(i):
        return i

    def upper_band(i):
        return 5 + i

    def lower_band(i):
        return 10 + i

    def bottom(i):
        return 15 + i

    connections = []
    for i in range(5):
        connections += [2, upper_band(i), lower_band(i), top(i)]
        if i < 4:
            connections += [2, lower_band(i), upper_band(i + 1), bottom(i)]
        else:
            connections += [1, lower_band(i), bottom(i)]

    # make_net_tforms narrates its walk over the tree on stdout.
    with contextlib.redirect_stdout(io.StringIO()):
        transforms = make_net_tforms(
            BASE_TRIANGLE_2D, connectivity, connections)
    return [np.asarray(transform, dtype=float) for transform in transforms]


def overlapping_faces(transforms):
    """
    Return how many net faces land on top of another one.

    A net that overlaps itself cannot be cut out, and an unfolding tree that
    looks reasonable can still produce one, so this is checked rather than
    assumed.

    :param transforms: the net transform of each face
    """
    placed = np.array([h_tform(BASE_TRIANGLE_2D, t) for t in transforms])

    def contains(point, triangle):
        a, b, c = triangle
        v0, v1, v2 = c - a, b - a, point - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        denominator = d00 * d11 - d01 * d01
        u = (d11 * d20 - d01 * d21) / denominator
        v = (d00 * d21 - d01 * d20) / denominator
        return u > 1e-6 and v > 1e-6 and u + v < 1 - 1e-6

    samples = [(0.25, 0.25), (0.5, 0.25), (0.25, 0.5), (1 / 3, 1 / 3)]
    overlapping = 0
    for i, face in enumerate(placed):
        points = [face[0] + u * (face[1] - face[0]) + v * (face[2] - face[0])
                  for u, v in samples]
        if any(contains(p, other) for j, other in enumerate(placed) if j != i
               for p in points):
            overlapping += 1
    return overlapping


def _row(values):
    """
    Format one row of numbers for a checked-in constant.

    Values a hair from zero are snapped to it.  The construction puts real
    zeros in these transforms -- a face centred on an axis, a net translation
    of none -- and floating point returns them as 1e-17, which would otherwise
    be pasted in as noise that reads like meaning.
    """
    snapped = [0.0 if abs(v) < 1e-12 else float(v) for v in values]
    return "[" + ", ".join(f"{v:.8f}" for v in snapped) + "]"


def print_transforms(transforms):
    """Print the face transforms as the checked-in constant."""
    print("ICO_TFORMS = [")
    for rvec, translation in transforms:
        print(f"    ({_row(rvec)}, {_row(translation)}),")
    print("]")


def print_net(transforms):
    """Print the net transforms as the checked-in constant."""
    print("ICO_NET_FORMS = [")
    for transform in transforms:
        print("    [" + ", ".join(_row(row) for row in transform) + "],")
    print("]")


def main():
    vertices, faces = icosahedron()

    edges = [np.linalg.norm(vertices[a] - vertices[b])
             for face in faces for a, b in combinations(face, 2)]
    unique_edges = {tuple(sorted((a, b)))
                    for face in faces for a, b in combinations(face, 2)}
    inward = sum(1 for face in faces
                 if np.cross(vertices[face[1]] - vertices[face[0]],
                             vertices[face[2]] - vertices[face[0]])
                 @ vertices[face].mean(0) <= 0)

    transforms, residual = face_transforms(vertices, faces)
    net = net_transforms(faces)

    print(f"# edge lengths {min(edges):.6f} to {max(edges):.6f}", file=sys.stderr)
    print(f"# {len(vertices)} vertices, {len(unique_edges)} edges, "
          f"{len(faces)} faces", file=sys.stderr)
    print(f"# inward wound faces: {inward}", file=sys.stderr)
    print(f"# largest face transform residual: {residual:.3e}", file=sys.stderr)
    print(f"# net faces overlapping another: {overlapping_faces(net)}",
          file=sys.stderr)

    print_transforms(transforms)
    print()
    print_net(net)


if __name__ == "__main__":
    main()
