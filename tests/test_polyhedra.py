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
    cells_touching_face,
    clip_lattice_to_face,
    clip_polygon_to_face,
    corners_inside_face,
    corners_within_cells,
    face_depths,
    inset_face,
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


# -- clipping a pattern that can be cut ---------------------------------------

def _square(x, y, side):
    return np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]])


def _area(polygon):
    x, y = np.asarray(polygon)[:, 0], np.asarray(polygon)[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@pytest.mark.parametrize("cells_across, whole, touching", [
    (6, 8, 24), (8, 18, 38), (12, 46, 78), (16, 90, 132), (20, 146, 200)])
def test_a_cut_lattice_covers_the_face_a_whole_one_leaves_half_of(
        cells_across, whole, touching):
    """
    What printing partial cells buys, pinned.

    Whole cells cover 51% of a triangular face at six cells to an edge and 84%
    at twenty; the cells that touch the face cover all of it, because the parts
    of them that hang over the edge are what gets clipped away.
    """
    assert len(clip_lattice_to_face(TRIANGLE_CORNERS, cells_across)) == whole
    cells = cells_touching_face(TRIANGLE_CORNERS, cells_across)
    assert len(cells) == touching

    covered = sum(_area(clip_polygon_to_face(
        _square(column / cells_across, row / cells_across, 1 / cells_across),
        TRIANGLE_CORNERS)) for column, row in cells)
    assert covered == pytest.approx(np.sqrt(3) / 4, rel=1e-9)


def test_the_cells_that_touch_a_face_include_every_cell_wholly_inside_it():
    whole = {tuple(cell) for cell in clip_lattice_to_face(TRIANGLE_CORNERS, 14)}
    touching = {tuple(cell) for cell in cells_touching_face(TRIANGLE_CORNERS, 14)}
    assert whole < touching


def test_a_cell_that_only_touches_a_face_along_its_edge_is_left_out():
    """A cell with no area inside the face would be printed as nothing."""
    cells = {tuple(cell) for cell in cells_touching_face(TRIANGLE_CORNERS, 8)}
    assert (0, -1) not in cells   # below the bottom edge, sharing it
    assert (-1, -1) not in cells  # sharing only the origin


def test_a_square_face_keeps_its_whole_lattice_either_way():
    square = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    assert len(cells_touching_face(square, 8)) == 64


def test_a_cut_lattice_finer_than_one_cell_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        cells_touching_face(TRIANGLE_CORNERS, 0)


def test_clipping_keeps_the_part_of_a_polygon_inside_the_face():
    clipped = clip_polygon_to_face(_square(0.9, 0.0, 0.2), TRIANGLE_CORNERS)
    assert len(clipped) == 3
    assert np.all(face_depths(clipped, TRIANGLE_CORNERS) > -1e-12)
    assert _area(clipped) < _area(_square(0.9, 0.0, 0.2))


def test_clipping_leaves_a_polygon_already_inside_the_face_alone():
    inside = _square(0.4, 0.1, 0.1)
    assert _area(clip_polygon_to_face(inside, TRIANGLE_CORNERS)) == pytest.approx(
        _area(inside))


def test_a_polygon_with_no_area_inside_the_face_clips_to_nothing():
    assert len(clip_polygon_to_face(_square(2.0, 2.0, 0.2), TRIANGLE_CORNERS)) == 0
    assert len(clip_polygon_to_face(_square(-0.2, -0.2, 0.2), TRIANGLE_CORNERS)) == 0


def test_how_deep_inside_a_face_a_point_is():
    """Measured to the nearest edge, so a triangle's centre is its inradius."""
    centre = TRIANGLE_CORNERS[:, :2].mean(axis=0)
    assert face_depths(centre[None], TRIANGLE_CORNERS)[0] == pytest.approx(
        1 / (2 * np.sqrt(3)))
    assert face_depths(TRIANGLE_CORNERS, TRIANGLE_CORNERS) == pytest.approx(0)
    assert face_depths(np.array([[0.5, -0.1]]), TRIANGLE_CORNERS)[0] < 0


def test_an_inset_face_is_the_face_pulled_in_from_every_edge():
    inset = inset_face(TRIANGLE_CORNERS, 0.05)
    assert face_depths(inset, TRIANGLE_CORNERS) == pytest.approx(0.05)
    # Still a triangle, and still the same one, only smaller.
    assert inset.shape == (3, 2)
    assert _area(inset) < np.sqrt(3) / 4


def test_insetting_a_face_away_to_nothing_is_refused():
    with pytest.raises(ValueError, match="shrinks it away"):
        inset_face(TRIANGLE_CORNERS, 0.5)


@pytest.mark.parametrize("cells_across, within, inside", [
    (6, 2, 20), (8, 8, 33), (12, 29, 69), (16, 65, 120), (20, 114, 184)])
def test_a_cut_lattice_offers_more_corners_than_a_whole_one(
        cells_across, within, inside):
    """
    The point of cutting the cells: corners the whole-cell clip cannot offer.

    A corner needs its four cells, and where one of them was dropped for
    hanging over the edge there is no corner to find.  Printed cut, the pattern
    reaches the edge and those corners are there -- counted here with no
    margin, so the face's own edge counts as inside it.
    """
    cells = clip_lattice_to_face(TRIANGLE_CORNERS, cells_across)
    assert len(corners_within_cells(cells)) == within
    assert len(corners_inside_face(TRIANGLE_CORNERS, cells_across)) == inside


def test_every_corner_the_whole_cell_clip_finds_is_also_inside_the_face():
    """Cutting the cells takes nothing away, whatever else it adds."""
    for cells_across in (6, 11, 16, 24):
        cells = clip_lattice_to_face(TRIANGLE_CORNERS, cells_across)
        within = {tuple(c) for c in corners_within_cells(cells)}
        inside = {tuple(c) for c in corners_inside_face(
            TRIANGLE_CORNERS, cells_across)}
        assert within <= inside


def test_asking_for_a_margin_keeps_only_the_corners_that_far_in():
    cells_across = 16
    pitch = 1 / cells_across
    for margin in (0.0, 0.35, 1.0):
        corners = corners_inside_face(TRIANGLE_CORNERS, cells_across, margin)
        depths = face_depths(corners * pitch, TRIANGLE_CORNERS)
        assert np.all(depths >= margin * pitch - 1e-12)
    assert (len(corners_inside_face(TRIANGLE_CORNERS, cells_across, 1.0))
            < len(corners_inside_face(TRIANGLE_CORNERS, cells_across, 0.0)))


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
