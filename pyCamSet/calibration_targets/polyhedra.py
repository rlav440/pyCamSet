"""
The solids a faced target is printed on.

A faced target is two separable things: a polyhedron, and a pattern drawn on
one of its faces.  The cube targets do not separate them -- ``ccube.py``
carries its own ``TFORMS`` and ``NET_FORMS``, ``puzzleboard_cube.py`` carries a
verbatim copy of both, and ``ccube2.py`` imports Ccube's -- so the same solid is
described three times.

This module is the other half: it owns solids and knows nothing about
patterns.  :class:`PolyhedralBasis` holds where each face sits on the solid,
where it sits in the printable net, and what shape it is; the three icosahedral
targets differ only in what they draw on the triangle it hands them.

It is written over a general polyhedron rather than over the icosahedron, so
that the cube can be moved onto it later, but only :func:`make_icosahedral` is
built today::

    basis = make_icosahedral()
    face_data = FaceToShape(
        face_local_coords=points,
        face_transforms=basis.face_matrices(),
        scale_factor=edge_length,
    )

The constants are generated offline by
``setup_scripts/calculate_ico_transforms.py`` and checked in, as the cube's
are: deriving them at import would pay numba's compilation cost every time
anything imported a target.  A test regenerates and compares them.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform

#: Height of a unit equilateral triangle.
TRIANGLE_HEIGHT = float(np.sqrt(3) / 2)

#: One icosahedral face, edge length one, wound anticlockwise seen from +z.
#: The z=0 plane every face pattern is drawn in.
TRIANGLE_CORNERS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, TRIANGLE_HEIGHT, 0.0],
])

#: Local face coordinates to icosahedron coordinates, as (rotation vector in
#: radians, translation in edge lengths).  Unitless, so one list describes an
#: icosahedron of any size -- :class:`FaceToShape` divides by the edge length
#: before transforming and multiplies after.
#:
#: Faces are numbered in four rings of five, from the top down: 0-4 the top
#: cap, 5-9 the upper band, 10-14 the lower band, 15-19 the bottom cap.
ICO_TFORMS = [
    ([-0.35545742, 0.54735643, 0.10097801], [0.00000000, 0.00000000, 0.95105652]),
    ([-0.65716619, 0.25226247, 1.30961330], [0.00000000, 0.00000000, 0.95105652]),
    ([-0.84480221, -0.22636407, 2.49748472], [0.00000000, 0.00000000, 0.95105652]),
    ([0.52499442, 0.64831411, -2.30240871], [0.00000000, 0.00000000, 0.95105652]),
    ([0.03604487, 0.68777715, -1.10891919], [0.00000000, 0.00000000, 0.95105652]),
    ([-0.07589120, 1.44809028, 0.95487549], [0.85065081, 0.00000000, 0.42532540]),
    ([-1.07752643, 1.33063432, 1.93265624], [0.26286556, 0.80901699, 0.42532540]),
    ([1.72140461, -0.46124898, -2.08127617], [-0.68819096, 0.50000000, 0.42532540]),
    ([1.38002205, 0.52974083, -1.12474182], [-0.68819096, -0.50000000, 0.42532540]),
    ([0.75303490, 1.15957206, -0.08748896], [0.26286556, -0.80901699, 0.42532540]),
    ([0.35640354, 2.25024339, 1.86115587], [0.68819096, 0.50000000, -0.42532540]),
    ([0.93359457, -1.83228251, -1.51546422], [-0.26286556, 0.80901699, -0.42532540]),
    ([1.61301809, -0.82187377, -0.67976433], [-0.85065081, 0.00000000, -0.42532540]),
    ([1.74334398, 0.27611856, 0.22837515], [-0.26286556, -0.80901699, -0.42532540]),
    ([1.34684948, 1.34684948, 1.11396697], [0.68819096, -0.50000000, -0.42532540]),
    ([-1.74716959, -2.15757476, -0.87673269], [0.00000000, 0.00000000, -0.95105652]),
    ([-0.13368969, -2.55095126, -0.47060737], [0.00000000, 0.00000000, -0.95105652]),
    ([1.35602673, -2.08809805, 0.04407698], [0.00000000, 0.00000000, -0.95105652]),
    ([2.40904969, -0.92474753, 0.54931066], [0.00000000, 0.00000000, -0.95105652]),
    ([2.73201605, 0.73204150, 0.92413506], [0.00000000, 0.00000000, -0.95105652]),
]

#: Local face coordinates to the printable net's coordinates, as 3x3 affines in
#: face-side units, x then y.  The ten band faces zigzag across, with each cap
#: face hung off its band face.
#:
#: Not :meth:`FaceToShape.draw_net`'s row/column convention, which the cube's
#: NET_FORMS use: an icosahedral net is drawn as vector polygons rather than by
#: blitting face images, because draw_net takes its canvas bounds from two of
#: each face image's four corners and composites subtractively over whole
#: rectangles -- and the bounding boxes of triangles at sixty degrees overlap,
#: so shared area would be darkened twice.
ICO_NET_FORMS = [
    [[0.50000000, 0.86602540, -0.50000000], [-0.86602540, 0.50000000, 0.86602540], [0.00000000, 0.00000000, 1.00000000]],
    [[0.50000000, 0.86602540, 0.00000000], [-0.86602540, 0.50000000, 1.73205081], [0.00000000, 0.00000000, 1.00000000]],
    [[0.50000000, 0.86602540, 0.50000000], [-0.86602540, 0.50000000, 2.59807621], [0.00000000, 0.00000000, 1.00000000]],
    [[0.50000000, 0.86602540, 1.00000000], [-0.86602540, 0.50000000, 3.46410162], [0.00000000, 0.00000000, 1.00000000]],
    [[0.50000000, 0.86602540, 1.50000000], [-0.86602540, 0.50000000, 4.33012702], [0.00000000, 0.00000000, 1.00000000]],
    [[1.00000000, 0.00000000, 0.00000000], [0.00000000, 1.00000000, 0.00000000], [0.00000000, 0.00000000, 1.00000000]],
    [[1.00000000, 0.00000000, 0.50000000], [0.00000000, 1.00000000, 0.86602540], [0.00000000, 0.00000000, 1.00000000]],
    [[1.00000000, 0.00000000, 1.00000000], [0.00000000, 1.00000000, 1.73205081], [0.00000000, 0.00000000, 1.00000000]],
    [[1.00000000, 0.00000000, 1.50000000], [0.00000000, 1.00000000, 2.59807621], [0.00000000, 0.00000000, 1.00000000]],
    [[1.00000000, 0.00000000, 2.00000000], [0.00000000, 1.00000000, 3.46410162], [0.00000000, 0.00000000, 1.00000000]],
    [[0.50000000, -0.86602540, 1.00000000], [0.86602540, 0.50000000, 0.00000000], [0.00000000, 0.00000000, 1.00000000]],
    [[0.50000000, -0.86602540, 1.50000000], [0.86602540, 0.50000000, 0.86602540], [0.00000000, 0.00000000, 1.00000000]],
    [[0.50000000, -0.86602540, 2.00000000], [0.86602540, 0.50000000, 1.73205081], [0.00000000, 0.00000000, 1.00000000]],
    [[0.50000000, -0.86602540, 2.50000000], [0.86602540, 0.50000000, 2.59807621], [0.00000000, 0.00000000, 1.00000000]],
    [[0.50000000, -0.86602540, 3.00000000], [0.86602540, 0.50000000, 3.46410162], [0.00000000, 0.00000000, 1.00000000]],
    [[-0.50000000, -0.86602540, 2.00000000], [0.86602540, -0.50000000, 0.00000000], [0.00000000, 0.00000000, 1.00000000]],
    [[-0.50000000, -0.86602540, 2.50000000], [0.86602540, -0.50000000, 0.86602540], [0.00000000, 0.00000000, 1.00000000]],
    [[-0.50000000, -0.86602540, 3.00000000], [0.86602540, -0.50000000, 1.73205081], [0.00000000, 0.00000000, 1.00000000]],
    [[-0.50000000, -0.86602540, 3.50000000], [0.86602540, -0.50000000, 2.59807621], [0.00000000, 0.00000000, 1.00000000]],
    [[-0.50000000, -0.86602540, 4.00000000], [0.86602540, -0.50000000, 3.46410162], [0.00000000, 0.00000000, 1.00000000]],
]

@dataclass(frozen=True)
class PolyhedralBasis:
    """
    A solid a target can be printed on.

    Holds where each face sits on the solid and in its printable net, in the
    same unitless edge-length convention the cube targets use, so one basis
    describes a family of solids of any size.

    :param name: what the solid is called, for messages and file names
    :param face_transforms: one ``(rotation vector, translation)`` per face,
        placing :attr:`base_face` on the solid.  Rotation is an OpenCV
        Rodrigues vector in radians; translation is in edge lengths.
    :param net_transforms: one 3x3 affine per face, in face-side units and in
        x/y order, placing the face in the printable net
    :param base_face: the corners of one face, edge length one, in its own z=0
        plane
    """

    name: str
    face_transforms: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]
    net_transforms: tuple[tuple[tuple[float, ...], ...], ...]
    base_face: np.ndarray

    @property
    def n_faces(self) -> int:
        """How many faces the solid has."""
        return len(self.face_transforms)

    @property
    def n_corners(self) -> int:
        """How many corners one face has."""
        return len(self.base_face)

    def face_matrices(self) -> list[np.ndarray]:
        """
        Return each face's placement as a 4x4 homogeneous transform.

        This is what :class:`FaceToShape` takes, and it expects the unitless
        form: pass the edge length as its ``scale_factor``.
        """
        return [make_4x4h_tform(np.asarray(rvec, dtype=float),
                                np.asarray(translation, dtype=float))
                for rvec, translation in self.face_transforms]

    def net_affine(self, face_index: int, edge_length: float) -> np.ndarray:
        """
        Return one face's net placement as an x/y affine.

        Only the translation is scaled, as the cube's ``net_affine_for_face``
        does: the face being placed is expected to already be drawn at the
        edge length asked for.

        :param face_index: which face
        :param edge_length: the solid's edge length, in whatever unit the
            caller draws in; the returned translation is in that unit
        """
        affine = np.asarray(self.net_transforms[face_index], dtype=float).copy()
        affine[:2, 2] *= float(edge_length)
        return affine

    def face_corners(self, edge_length: float = 1.0) -> np.ndarray:
        """
        Return every face's corners in the solid's own coordinates.

        :param edge_length: the solid's edge length
        :return: ``(n_faces, n_corners, 3)``
        """
        base = self.base_face / 1.0
        return np.array([
            h_tform(base, matrix) * float(edge_length)
            for matrix in self.face_matrices()
        ])

    def face_normals(self) -> np.ndarray:
        """Return each face's outward unit normal."""
        centres = self.face_corners().mean(axis=1)
        return centres / np.linalg.norm(centres, axis=1, keepdims=True)

    def adjacent_faces(self) -> tuple[tuple[int, int], ...]:
        """
        Return the face pairs that share an edge.

        Derived from :attr:`face_transforms` rather than written down, so it
        cannot drift from the geometry it describes.
        """
        corners = self.face_corners()
        pairs = []
        for i, j in combinations(range(self.n_faces), 2):
            shared = sum(
                1 for a in corners[i]
                if np.min(np.linalg.norm(corners[j] - a, axis=1)) < 1e-6)
            if shared == 2:
                pairs.append((i, j))
        return tuple(pairs)

    def antipodal_faces(self) -> tuple[tuple[int, int], ...]:
        """
        Return the face pairs that point opposite ways.

        These are the pairs a camera can never see at once, whatever it does,
        so they are the pairs whose patterns need not be told apart.
        """
        normals = self.face_normals()
        return tuple(
            (i, j) for i, j in combinations(range(self.n_faces), 2)
            if normals[i] @ normals[j] < -1 + 1e-9)

    def covisible(self, face: int, other: int) -> bool:
        """
        Whether some viewpoint sees both faces at once.

        On a convex solid a distant camera sees exactly the faces whose
        outward normal points towards it, so two faces can be seen together
        unless they point in opposite directions.

        :param face: one face index
        :param other: the other face index
        """
        normals = self.face_normals()
        return bool(normals[face] @ normals[other] > -1 + 1e-9)

    def solid(self, edge_length: float = 1.0):
        """
        Return the solid as a shared-vertex mesh.

        Built from :attr:`face_transforms`, not from an independent description
        of the same solid: anything else could disagree with the geometry the
        target calibrates against, and a printed core that disagrees is a
        mistake nothing would catch until it had been made.

        :param edge_length: the solid's edge length
        :return: the vertices, and the vertex indices of each face
        """
        placed = self.face_corners(edge_length)
        vertices: list[np.ndarray] = []
        faces = []
        # Loose next to an edge, but far tighter than any two distinct corners
        # come: the transforms are checked in rounded to eight places, so
        # corners that meet exactly still arrive about 1e-8 apart.
        tolerance = 1e-6 * float(edge_length)
        for face in placed:
            indices = []
            for corner in face:
                if vertices:
                    distances = np.linalg.norm(
                        np.asarray(vertices) - corner, axis=1)
                    match = np.flatnonzero(distances < tolerance)
                    if match.size:
                        indices.append(int(match[0]))
                        continue
                vertices.append(corner)
                indices.append(len(vertices) - 1)
            faces.append(indices)
        return np.array(vertices), np.array(faces)

    def to_stl(self, f_out: Path | str, edge_length: float = 100.0) -> Path:
        """
        Write the bare solid as an STL, to print a core to mount faces on.

        Folding an accurate twenty-faced net by hand is hard, and the printed
        pattern is only as good as the solid under it.  This writes the solid
        itself, at true size and in millimetres, for printing.

        :param f_out: where to write it
        :param edge_length: the solid's edge length in millimetres
        :raises ValueError: for an edge length that is not positive
        """
        if edge_length <= 0:
            raise ValueError("edge_length must be greater than zero.")
        try:
            import pyvista as pv
        except ImportError as error:
            raise ImportError(
                "Writing a target's solid as an STL needs pyvista, which is "
                "not installed. Install pyvista to export one.") from error

        f_out = Path(f_out).expanduser().with_suffix(".stl").resolve()
        f_out.parent.mkdir(parents=True, exist_ok=True)

        vertices, faces = self.solid(edge_length)
        # An STL holds triangles only. An icosahedron's faces already are
        # triangles; a cube's would not be, so this is not left to chance.
        cells = np.hstack([[len(face), *face] for face in faces])
        mesh = pv.PolyData(vertices, faces=cells).triangulate()
        mesh.save(str(f_out))
        return f_out


def make_icosahedral() -> PolyhedralBasis:
    """
    Return the icosahedron twenty triangular faces are printed on.

    Twenty faces rather than six is the point of it: exactly ten of them face
    any distant viewpoint, against a cube's three, so many more faces are
    visible to a rig at once and the orientations a target presents are much
    more finely spaced.
    """
    return PolyhedralBasis(
        name="icosahedron",
        face_transforms=tuple(
            (tuple(rvec), tuple(translation))
            for rvec, translation in ICO_TFORMS),
        net_transforms=tuple(
            tuple(tuple(value) for value in matrix)
            for matrix in ICO_NET_FORMS),
        base_face=TRIANGLE_CORNERS.copy(),
    )


def clip_lattice_to_face(
    face: np.ndarray,
    cells_across: float,
    phase: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Return the square lattice cells lying wholly inside a face.

    A square lattice does not fit a triangle, and a marker is built of whole
    cells: half a cell carries half a marker, which reads as nothing.  Clipping
    to whole cells keeps 51% of a triangular face at six cells to an edge and
    84% at twenty, against the 50% the largest rectangle that fits inside a
    triangle would keep at any size.  A pattern that *can* be cut -- a
    chessboard, whose cut cells still meet their neighbours in corners -- keeps
    the whole face instead, with :func:`cells_touching_face`.  Shifting ``phase`` by
    a fraction of a cell is worth another cell or two at coarse pitches, and
    ``cells_across`` need not be a whole number: letting the lattice fall a
    fraction of a cell short of the edge buys a little more again.
    :class:`~pyCamSet.calibration_targets.cico2.CIco2` chooses both.

    Cells that are dropped are simply not printed.  Nothing downstream needs
    telling: a detector reading a board whose cells are missing is reading a
    partly hidden board, which is a thing every detector here already handles.

    :param face: the face's corners as ``(k, 2)`` or ``(k, 3)``, convex and
        wound anticlockwise, in edge lengths
    :param cells_across: how many cells span one edge length, which need not
        be a whole number of them
    :param phase: where the lattice starts, in cells, as ``(x, y)``
    :return: the ``(n, 2)`` integer ``(column, row)`` index of each whole cell,
        in row-major order
    :raises ValueError: for a cell count that is not positive
    """
    if not float(cells_across) >= 1:
        raise ValueError("cells_across must be at least one.")
    polygon = np.asarray(face, dtype=float)[:, :2]
    pitch = 1.0 / float(cells_across)

    # A convex polygon wound anticlockwise holds every point that is left of
    # each of its edges, so a cell is whole exactly when all four of its
    # corners are -- there is no need to look at the cell's interior.
    edges = np.roll(polygon, -1, axis=0) - polygon

    def inside(points):
        offsets = points[:, None, :] - polygon[None, :, :]
        cross = (edges[None, :, 0] * offsets[..., 1]
                 - edges[None, :, 1] * offsets[..., 0])
        return np.all(cross > -1e-12, axis=1)

    low = np.floor(polygon.min(axis=0) / pitch).astype(int) - 1
    high = np.ceil(polygon.max(axis=0) / pitch).astype(int) + 1
    columns = np.arange(low[0], high[0] + 1)
    rows = np.arange(low[1], high[1] + 1)
    grid = np.stack(np.meshgrid(columns, rows, indexing="xy"), axis=-1)
    grid = grid.reshape(-1, 2)

    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    whole = np.ones(len(grid), dtype=bool)
    for corner in corners:
        points = (grid + phase + corner) * pitch
        whole &= inside(points)
    return grid[whole]


def corners_within_cells(cells: np.ndarray) -> np.ndarray:
    """
    Return the lattice corners every one of whose four cells is present.

    The feature a chessboard corner is is the meeting of four cells, two
    black and two white.  Where one of the four was clipped away there is no
    corner to find, only the end of a line, so these are the corners a face
    printed in whole cells can actually offer -- and the ones a target should
    keep object points for.  A face printed with cut cells has more of them, and
    asks :func:`corners_inside_face` instead.

    :param cells: the ``(n, 2)`` cell indices :func:`clip_lattice_to_face`
        returned
    :return: the ``(m, 2)`` corner indices, in row-major order.  Corner
        ``(i, j)`` is the point at ``(i, j)`` cell widths from the lattice
        origin, shared by cells ``(i-1, j-1)`` through ``(i, j)``.
    """
    present = {(int(column), int(row)) for column, row in cells}
    corners = sorted(
        {(column + dx, row + dy)
         for column, row in present for dx in (0, 1) for dy in (0, 1)},
        key=lambda corner: (corner[1], corner[0]))
    kept = [
        corner for corner in corners
        if all((corner[0] + dx, corner[1] + dy) in present
               for dx in (-1, 0) for dy in (-1, 0))
    ]
    return np.array(kept, dtype=int).reshape(-1, 2)


def face_depths(points: np.ndarray, face: np.ndarray) -> np.ndarray:
    """
    Return how far inside a face each point lies, in the face's own units.

    Measured to the nearest of the face's edge lines, which for a point inside
    a convex face is its distance to the face's boundary, and which goes
    negative outside the face.

    :param points: the points as ``(n, 2)`` or ``(n, 3)``
    :param face: the face's corners as ``(k, 2)`` or ``(k, 3)``, convex and
        wound anticlockwise
    :return: the ``(n,)`` depth of each point
    """
    polygon = np.asarray(face, dtype=float)[:, :2]
    points = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = points[:, None, :] - polygon[None, :, :]
    cross = (edges[None, :, 0] * offsets[..., 1]
             - edges[None, :, 1] * offsets[..., 0])
    return np.min(cross / np.linalg.norm(edges, axis=1)[None, :], axis=1)


def inset_face(face: np.ndarray, distance: float) -> np.ndarray:
    """
    Return a face shrunk by moving every one of its edges inwards.

    :param face: the face's corners as ``(k, 2)`` or ``(k, 3)``, convex and
        wound anticlockwise
    :param distance: how far each edge moves, in the face's own units
    :return: the shrunk face's ``(k, 2)`` corners
    :raises ValueError: for a distance that shrinks the face away entirely
    """
    polygon = np.asarray(face, dtype=float)[:, :2]
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    offsets = np.sum(normals * polygon, axis=1) + float(distance)

    # A corner of the shrunk face is where its two moved edges now cross.
    corners = np.stack([
        np.linalg.solve(np.stack([normals[i - 1], normals[i]]),
                        np.array([offsets[i - 1], offsets[i]]))
        for i in range(len(polygon))
    ])
    if np.any(face_depths(corners, polygon) < -1e-12):
        raise ValueError(
            f"Insetting this face by {distance} shrinks it away entirely.")
    return corners


def clip_polygon_to_face(polygon: np.ndarray, face: np.ndarray) -> np.ndarray:
    """
    Return the part of a polygon that lies inside a face.

    Sutherland-Hodgman, which is exact for the convex faces here: the polygon
    is cut against one edge of the face at a time.

    :param polygon: the polygon's corners as ``(n, 2)``, wound anticlockwise
    :param face: the face's corners as ``(k, 2)`` or ``(k, 3)``, convex and
        wound anticlockwise
    :return: the clipped polygon's ``(m, 2)`` corners, empty where the part
        inside the face has no area
    """
    kept = [point for point in np.asarray(polygon, dtype=float)[:, :2]]
    boundary = np.asarray(face, dtype=float)[:, :2]
    for index in range(len(boundary)):
        start = boundary[index]
        edge = boundary[(index + 1) % len(boundary)] - start
        sides = [float(edge[0] * (point[1] - start[1])
                       - edge[1] * (point[0] - start[0])) for point in kept]
        clipped = []
        for i, point in enumerate(kept):
            j = (i + 1) % len(kept)
            here, there = sides[i], sides[j]
            if here >= -1e-15:
                clipped.append(point)
            if (here > 1e-15) != (there > 1e-15):
                clipped.append(point + (kept[j] - point) * (here / (here - there)))
        kept = clipped
        if not kept:
            return np.empty((0, 2))

    # Cutting through a corner of the polygon lands the same point in the list
    # twice, which is a corner the shape does not have.
    distinct = [point for index, point in enumerate(kept)
                if not np.allclose(point, kept[index - 1], atol=1e-12)]
    if len(distinct) < 3:
        return np.empty((0, 2))
    return np.array(distinct)


def cells_touching_face(
    face: np.ndarray,
    cells_across: float,
    phase: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Return the square lattice cells with any of their area inside a face.

    The counterpart of :func:`clip_lattice_to_face`, for a pattern that can be
    printed as a part of a cell: a chessboard can, because a cut square still
    meets its neighbours in corners, whereas a marker cannot.  Where the
    pattern allows it this keeps the whole face rather than the 51-84% whole
    cells do, and the cells it adds are drawn clipped -- see
    :func:`clip_polygon_to_face`.

    :param face: the face's corners as ``(k, 2)`` or ``(k, 3)``, convex and
        wound anticlockwise, in edge lengths
    :param cells_across: how many cells span one edge length, which need not
        be a whole number of them
    :param phase: where the lattice starts, in cells, as ``(x, y)``
    :return: the ``(n, 2)`` integer ``(column, row)`` index of each cell, in
        row-major order
    :raises ValueError: for a cell count that is not positive
    """
    if not float(cells_across) >= 1:
        raise ValueError("cells_across must be at least one.")
    polygon = np.asarray(face, dtype=float)[:, :2]
    pitch = 1.0 / float(cells_across)
    edges = np.roll(polygon, -1, axis=0) - polygon

    low = np.floor(polygon.min(axis=0) / pitch).astype(int) - 1
    high = np.ceil(polygon.max(axis=0) / pitch).astype(int) + 1
    grid = np.stack(np.meshgrid(np.arange(low[0], high[0] + 1),
                                np.arange(low[1], high[1] + 1),
                                indexing="xy"), axis=-1).reshape(-1, 2)

    # Two convex shapes are apart only if one of their own edges holds them
    # apart, so a cell overlaps the face unless some face edge leaves all four
    # of the cell's corners outside, or the cell misses the face's box.
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    points = (grid[:, None, :] + np.asarray(phase, dtype=float)
              + corners[None, :, :]) * pitch
    offsets = points[:, :, None, :] - polygon[None, None, :, :]
    cross = (edges[None, None, :, 0] * offsets[..., 1]
             - edges[None, None, :, 1] * offsets[..., 0])
    held_apart = np.any(np.all(cross < 1e-12, axis=1), axis=1)
    in_box = np.all((points.max(axis=1) > polygon.min(axis=0) + 1e-12)
                    & (points.min(axis=1) < polygon.max(axis=0) - 1e-12), axis=1)
    return grid[in_box & ~held_apart]


def corners_inside_face(
    face: np.ndarray,
    cells_across: float,
    margin: float = 0.0,
    phase: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Return the lattice corners lying at least ``margin`` cells inside a face.

    The counterpart of :func:`corners_within_cells` for a face printed with
    partial cells.  Where whole cells are printed a corner exists only where
    all four of its cells were kept; where partial ones are, the pattern
    reaches the face's edge, and a corner is a corner wherever enough of that
    pattern surrounds it.  ``margin`` is that "enough": a corner on the edge of
    what was printed is half a junction and is not found, so the caller asks
    for the collar its detector needs.

    :param face: the face's corners as ``(k, 2)`` or ``(k, 3)``, convex and
        wound anticlockwise, in edge lengths
    :param cells_across: how many cells span one edge length
    :param margin: how far inside the face a corner must be, in cells
    :param phase: where the lattice starts, in cells, as ``(x, y)``
    :return: the ``(n, 2)`` integer corner indices, in row-major order, in
        :func:`corners_within_cells`' convention
    :raises ValueError: for a cell count that is not positive
    """
    if not float(cells_across) >= 1:
        raise ValueError("cells_across must be at least one.")
    polygon = np.asarray(face, dtype=float)[:, :2]
    pitch = 1.0 / float(cells_across)

    low = np.floor(polygon.min(axis=0) / pitch).astype(int) - 1
    high = np.ceil(polygon.max(axis=0) / pitch).astype(int) + 1
    grid = np.stack(np.meshgrid(np.arange(low[0], high[0] + 1),
                                np.arange(low[1], high[1] + 1),
                                indexing="xy"), axis=-1).reshape(-1, 2)
    points = (grid + np.asarray(phase, dtype=float)) * pitch
    deep = face_depths(points, polygon) >= float(margin) * pitch - 1e-12
    return grid[deep]
