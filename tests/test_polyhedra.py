"""
What the solids a faced target is printed on are.

The constants in :mod:`pyCamSet.calibration_targets.polyhedra` are generated
offline and pasted in, which is the convention the cube targets already set but
which leaves nothing checking that what was pasted is what was generated.  The
first test here is that check.  The rest are the properties a target relies on
without ever asserting them: that the solid closes up, that its faces are
numbered the way the module says, and that the net can be cut out.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from pyCamSet.calibration_targets.polyhedra import (
    ICO_NET_FORMS,
    ICO_TFORMS,
    TRIANGLE_CORNERS,
    clip_lattice_to_face,
    make_icosahedral,
)
from pyCamSet.utils.general_utils import h_tform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "setup_scripts"))


@pytest.fixture(scope="module")
def basis():
    return make_icosahedral()


# -- the constants are what the generator makes -------------------------------

def test_the_checked_in_constants_are_what_the_generator_makes():
    """
    A pasted constant can drift from the tool that made it.

    make_target_net.py's own docstring warns that a fresh run permutes two of
    the cube's faces, so this is not a theoretical worry.
    """
    from calculate_ico_transforms import (
        face_transforms, icosahedron, net_transforms)

    vertices, faces = icosahedron()
    transforms, _ = face_transforms(vertices, faces)

    assert len(transforms) == len(ICO_TFORMS)
    for (rvec, translation), (checked_r, checked_t) in zip(
            transforms, ICO_TFORMS):
        assert np.allclose(rvec, checked_r, atol=1e-8)
        assert np.allclose(translation, checked_t, atol=1e-8)

    net = net_transforms(faces)
    assert len(net) == len(ICO_NET_FORMS)
    for made, checked in zip(net, ICO_NET_FORMS):
        assert np.allclose(made, checked, atol=1e-8)


# -- the solid ----------------------------------------------------------------

def test_the_faces_close_up_into_an_icosahedron(basis):
    vertices, faces = basis.solid(1.0)
    assert len(vertices) == 12
    assert len(faces) == 20

    edges = [np.linalg.norm(vertices[a] - vertices[b])
             for face in faces for a, b in combinations(face, 2)]
    assert np.allclose(edges, 1.0, atol=1e-6)

    unique = {tuple(sorted((a, b)))
              for face in faces for a, b in combinations(face, 2)}
    assert len(unique) == 30
    # Euler's formula, which only a closed solid satisfies.
    assert len(vertices) - len(unique) + len(faces) == 2


def test_every_face_is_wound_so_its_normal_points_outwards(basis):
    vertices, faces = basis.solid(1.0)
    for face in faces:
        corners = vertices[face]
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        assert normal @ corners.mean(axis=0) > 0


def test_the_faces_are_numbered_in_four_rings_of_five(basis):
    """The module documents 0-4 top cap, 5-9 and 10-14 the bands, 15-19 bottom."""
    heights = basis.face_corners().mean(axis=1)[:, 2]
    rings = [np.round(heights[5 * i:5 * (i + 1)], 6) for i in range(4)]
    for ring in rings:
        assert len(set(ring.tolist())) == 1
    centres = [ring[0] for ring in rings]
    assert centres == sorted(centres, reverse=True)


def test_a_scaled_solid_scales_its_edges(basis):
    vertices, faces = basis.solid(37.5)
    edges = [np.linalg.norm(vertices[a] - vertices[b])
             for face in faces for a, b in combinations(face, 2)]
    assert np.allclose(edges, 37.5, rtol=1e-6)


# -- which faces can be seen together -----------------------------------------

def test_thirty_pairs_of_faces_share_an_edge(basis):
    assert len(basis.adjacent_faces()) == 30


def test_ten_pairs_of_faces_point_opposite_ways(basis):
    antipodal = basis.antipodal_faces()
    assert len(antipodal) == 10
    # Every face is in exactly one such pair.
    assert sorted(f for pair in antipodal for f in pair) == list(range(20))


def test_a_distant_camera_sees_exactly_ten_faces(basis):
    """
    The reason for preferring an icosahedron to a cube, asserted rather than
    claimed.  A cube shows three faces; this shows ten, from anywhere.
    """
    normals = basis.face_normals()
    generator = np.random.default_rng(0)
    directions = generator.normal(size=(500, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    counts = {int(np.sum(normals @ d > 0)) for d in directions}
    assert counts == {10}


def test_faces_are_covisible_unless_they_are_antipodal(basis):
    antipodal = set(basis.antipodal_faces())
    for i, j in combinations(range(20), 2):
        assert basis.covisible(i, j) is ((i, j) not in antipodal)


# -- the net ------------------------------------------------------------------

def test_the_net_lays_every_face_out_without_overlapping(basis):
    """A net that overlaps itself cannot be cut out."""
    placed = np.array([
        h_tform(basis.base_face[:, :2], basis.net_affine(i, 1.0))
        for i in range(basis.n_faces)])
    assert placed.shape == (20, 3, 2)

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
    for i, face in enumerate(placed):
        points = [face[0] + u * (face[1] - face[0]) + v * (face[2] - face[0])
                  for u, v in samples]
        for j, other in enumerate(placed):
            if i == j:
                continue
            assert not any(contains(p, other) for p in points), (
                f"net face {i} overlaps face {j}")


def test_the_net_keeps_every_face_the_size_it_is(basis):
    """Unfolding is rigid: a face in the net is the face on the solid."""
    for i in range(basis.n_faces):
        placed = h_tform(basis.base_face[:, :2], basis.net_affine(i, 1.0))
        edges = [np.linalg.norm(placed[a] - placed[b])
                 for a, b in combinations(range(3), 2)]
        assert np.allclose(edges, 1.0, atol=1e-6)


def test_the_net_scales_with_the_edge_length(basis):
    """
    A net affine scales its translation only: the face handed to it is
    expected to already be drawn at the edge length asked for.
    """
    face = basis.base_face[:, :2]
    small = h_tform(face, basis.net_affine(7, 1.0))
    large = h_tform(face * 10.0, basis.net_affine(7, 10.0))
    assert np.allclose(large, small * 10.0, atol=1e-6)


# -- clipping a lattice to a face ---------------------------------------------

@pytest.mark.parametrize("cells_across, expected", [
    (6, 8), (8, 18), (10, 30), (12, 46), (16, 90), (20, 146)])
def test_how_much_of_a_triangle_a_square_lattice_fills(cells_across, expected):
    """
    The clipped cell count, pinned.

    The fraction of the face these cover is the argument for clipping at all:
    the largest rectangle that fits inside a triangle covers half of it, at any
    size, and these are 51% at six cells to an edge rising to 84% at twenty.
    """
    cells = clip_lattice_to_face(TRIANGLE_CORNERS, cells_across)
    assert len(cells) == expected

    area = len(cells) * (1 / cells_across) ** 2
    assert area / (np.sqrt(3) / 4) > 0.5


def test_every_clipped_cell_lies_wholly_inside_the_face():
    cells_across = 12
    cells = clip_lattice_to_face(TRIANGLE_CORNERS, cells_across)
    pitch = 1 / cells_across
    height = np.sqrt(3) / 2
    for column, row in cells:
        for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1)):
            x, y = (column + dx) * pitch, (row + dy) * pitch
            assert y > -1e-9
            assert y < np.sqrt(3) * x + 1e-9
            assert y < np.sqrt(3) * (1 - x) + 1e-9
            assert y < height + 1e-9


def test_a_square_face_keeps_its_whole_lattice():
    """The clip is general over the face, and a square face loses nothing."""
    square = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    cells = clip_lattice_to_face(square, 8)
    assert len(cells) == 64


def test_shifting_the_lattice_can_fit_another_cell():
    flush = len(clip_lattice_to_face(TRIANGLE_CORNERS, 6))
    shifted = max(
        len(clip_lattice_to_face(TRIANGLE_CORNERS, 6, phase=(dx, dy)))
        for dx in np.linspace(0, 1, 9) for dy in np.linspace(0, 1, 9))
    assert shifted >= flush


def test_a_lattice_finer_than_one_cell_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        clip_lattice_to_face(TRIANGLE_CORNERS, 0)


# -- printing the solid -------------------------------------------------------

def test_the_solid_writes_itself_as_an_stl(basis, tmp_path):
    pv = pytest.importorskip("pyvista")
    written = basis.to_stl(tmp_path / "ico", edge_length=100.0)
    assert written.exists()
    assert written.suffix == ".stl"

    mesh = pv.read(str(written))
    assert mesh.n_points == 12
    assert mesh.n_faces == 20
    distances = np.linalg.norm(
        mesh.points[:, None] - mesh.points[None, :], axis=-1)
    assert np.isclose(distances[distances > 1e-3].min(), 100.0, rtol=1e-5)


def test_the_written_solid_is_the_one_the_target_calibrates_against(
        basis, tmp_path):
    """
    The printed core has to be the solid the face transforms describe.

    A core built from an independent description of an icosahedron could
    disagree by a rotation of the face numbering, and nothing would catch it
    until the target had been printed, assembled and calibrated against.
    """
    pv = pytest.importorskip("pyvista")
    mesh = pv.read(str(basis.to_stl(tmp_path / "ico", edge_length=50.0)))

    written = np.sort(mesh.cell_centers().points, axis=0)
    expected = np.sort(basis.face_corners(50.0).mean(axis=1), axis=0)
    assert np.allclose(written, expected, atol=1e-4)


def test_a_solid_with_no_size_is_refused(basis, tmp_path):
    with pytest.raises(ValueError, match="greater than zero"):
        basis.to_stl(tmp_path / "ico", edge_length=0.0)
