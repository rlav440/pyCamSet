"""
An icosahedron whose twenty triangular faces each carry a ChArUco2 board.

The cube version is :class:`~pyCamSet.calibration_targets.ccube2.Ccube2`, and
the design is aruco2's ``GridBoard``: a whole marker on every square, which is
what lets a board report the corners on its *outer* boundary as well as its
interior ones.  That matters a great deal here.  Clipping a board to a triangle
throws away its outside, and a design whose corners are the meetings of four
printed squares would lose most of what is left; one whose corners are the
corners of each printed square keeps them.  At eight squares to an edge a
clipped face holds 18 squares, which give 8 corners the first way and 30 the
second.

The cost is the marker alphabet, and clipping makes it worse before it makes
it better.  Every square carries its own marker, and every square of the
*rectangle* the board is cut from needs an id whether or not it is printed,
because the board that reads a face is that rectangle.  Given a block each,
twenty faces would spend more than half the alphabet on squares the triangle
cut away -- 22 of every 40 at eight squares to an edge -- and a
thousand-marker dictionary would run out there.

So the squares that are never printed do not get a block each.  They share one
block of filler ids between all twenty faces, which is safe precisely because
they are never printed: no image can contain one, so no two faces can be
confused by one.  A single repeated filler would be simpler still, and aruco2
refuses it -- a board must have a distinct id per square -- but a shared
*block* satisfies that while costing one alphabet rather than twenty.

It buys most of the target's resolution.  Twelve squares to an edge needs 2160
ids a block at a time and 982 shared, so it fits a thousand-marker dictionary
where it did not before: 66 corners a face rather than 30.

Where the lattice sits inside the triangle is free, and chosen rather than
left flush in a corner: see :data:`FACE_LATTICE`.  It is the alphabet that
decides how far that can be taken, so the two are settled together -- the
twelve-square face has a placement worth 68 corners, and it is not drawn
because it needs 1032 markers.

The band is the other thing clipping disturbs.  aruco2 draws a ring of
alternating tabs around a rectangular board, which is what gives its outer
corners a black and white side each; a clipped board's outer boundary is a
staircase instead, so the same rule -- a tab outward wherever a printed square
with ``(column + row)`` odd meets one that was clipped away -- is applied along
that staircase.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import svgwrite

from pyCamSet.calibration_targets.charuco2 import _EXCLUDED_DICT_PREFIXES
from pyCamSet.calibration_targets.core import (
    AbstractTarget, FaceToShape, ImageDetection, exclude_by_prefix,
)
from pyCamSet.calibration_targets.core.abstract_target import (
    EXPORT_SUFFIXES, export_path,
)
from pyCamSet.calibration_targets.core.parameters import (
    DocumentedParameters,
    Parameterisation,
)
from pyCamSet.calibration_targets.markers.aruco2 import _as_uint8_image
from pyCamSet.calibration_targets.markers.aruco2_gridboard import (
    CHARUCO2_DETECTOR,
    detect_grid_board_corners,
    dictionary_marker_count,
    grid_board_marker_bits,
    refuse_rotation_ambiguous_markers,
    warn_known_false_detections,
)
from pyCamSet.calibration_targets.markers.backend_registry import (
    ARUCO2_BACKEND,
    dict_names_for_backend,
    dictionary_id,
)
from pyCamSet.calibration_targets.markers.gridboard_layout import (
    band_depth,
    grid_board_corners,
    grid_board_rectangles,
)
from pyCamSet.calibration_targets.polyhedra import (
    TRIANGLE_HEIGHT,
    clip_lattice_to_face,
    make_icosahedral,
)
from pyCamSet.cameras import Camera

logger = logging.getLogger(__name__)

#: The smallest face worth clipping a board to.
_MIN_POINTS = 6

#: Where each face size draws its lattice: how many cells actually span the
#: face's bottom edge, and how far along it they start, in cells.
#:
#: A square lattice does not fit a triangle, so neither number is forced --
#: the lattice is clipped to whatever falls inside -- and both are worth
#: choosing.  Shifting the lattice along the edge is worth a corner or two,
#: and letting it fall a fraction of a cell short of the edge makes every
#: square about 1-3% larger for the same corners.  Together they are worth
#: 18 corners rather than 16 at six squares to an edge, and 66 rather than
#: 65 at twelve, with no more of the alphabet spent.
#:
#: Only the y shift is missing, and only because it is never useful: a face's
#: bottom edge lies along y = 0, so shifting off it can only spoil the row
#: that is flush with it.
#:
#: The sizes are those a dictionary in ordinary use can hold, which is what
#: the search was held to; past twelve a finer lattice cannot be afforded and
#: the best that fits is the twelve-square face again.  Generated by
#: setup_scripts/calculate_cico2_lattice.py and checked against it by
#: tests/test_cico2_target.py.
FACE_LATTICE: dict[int, tuple[float, float]] = {
    6: (5.906, 0.318),
    7: (6.794, 0.896),
    8: (7.794, 0.896),
    9: (8.794, 0.896),
    10: (9.906, 0.318),
    11: (10.908, 0.164),
    12: (11.794, 0.206),
}

#: The dictionary an icosahedron is printed with unless another is asked for.
#: Every printed square of every face carries its own marker, so it needs
#: twenty faces' worth of those, plus one shared block for the rest.
_DEFAULT_DICT_NAME = "DICT_4X4_1000"


def corners_touching_cells(cells: np.ndarray) -> np.ndarray:
    """
    Return every lattice corner that at least one cell reaches.

    The corners a ChArUco2 board offers, as against
    :func:`~pyCamSet.calibration_targets.polyhedra.corners_within_cells`,
    which is the ChArUco1 rule.  A marker on every square localises that
    square's own four corners, boundary ones included, so a corner survives
    clipping if any one of its squares was printed.

    :param cells: the ``(n, 2)`` cell indices that were printed
    :return: the ``(m, 2)`` corner indices, in row-major order
    """
    present = {(int(column), int(row)) for column, row in cells}
    corners = sorted(
        {(column + dx, row + dy)
         for column, row in present for dx in (0, 1) for dy in (0, 1)},
        key=lambda corner: (corner[1], corner[0]))
    return np.array(corners, dtype=int).reshape(-1, 2)


class CIco2(AbstractTarget):
    """An icosahedron of twenty triangular ChArUco2 faces."""

    DETECTOR_BACKENDS = {ARUCO2_BACKEND: CHARUCO2_DETECTOR}

    @classmethod
    def construction_parameters(cls, backend: str | None = None) -> Parameterisation:
        """What decides where an icosahedron's corners are."""
        return DocumentedParameters(
            cls.__init__,
            "n_points", "length", "border_fraction", "aruco_dict",
            choices={
                "aruco_dict": exclude_by_prefix(
                    dict_names_for_backend(ARUCO2_BACKEND),
                    *_EXCLUDED_DICT_PREFIXES,
                )
            },
        )

    @classmethod
    def export_parameters(cls) -> Parameterisation:
        """How a CIco2 net is drawn, which is not what it is."""
        return DocumentedParameters(
            cls.save_printable, "border_width", "draw_cut_outline",
            "draw_face_ids", "dpi")

    @classmethod
    def printable_name(cls, values: dict, kind: str = "svg") -> str:
        return (f"cico2_{int(values['n_points'])}points_"
                f"{float(values['length']):g}mm{EXPORT_SUFFIXES[kind]}")

    def save_printable(self, path, kind: str = "svg", border_width: float = 10.0,
                       draw_cut_outline: bool = True, draw_face_ids: bool = True,
                       dpi: int = 300) -> Path:
        """
        Write this icosahedron as a net to print.

        :param path: where to write it
        :param kind: one of :data:`EXPORT_KINDS`
        :param border_width: Net border (mm) -- the margin drawn around the
            folded net. Suggested: 10.
        :param draw_cut_outline: Draw cut outline -- an outline to cut the net
            out along. Suggested: on.
        :param draw_face_ids: Draw face numbers -- a number on each face, for
            folding it the right way up. Suggested: on.
        :param dpi: Raster DPI -- the resolution a raster PDF is rendered at.
            Ignored by the vector formats. Suggested: 300-600.
        :raises ValueError: for a format an icosahedron cannot be written as
        """
        if kind == "svg":
            return self.save_to_svg(
                path, border_width=border_width,
                draw_cut_outline=draw_cut_outline,
                draw_face_ids=draw_face_ids)
        if kind in ("pdf_vector", "pdf_raster"):
            return self.save_to_pdf(
                path, data_format="vector" if kind == "pdf_vector" else "raster",
                dpi=int(dpi), border_width=border_width,
                draw_cut_outline=draw_cut_outline,
                draw_face_ids=draw_face_ids)
        raise ValueError(f"A CIco2 cannot be written as {kind!r}.")

    def __init__(
        self,
        length: float = 100.0,
        n_points: int = 12,
        aruco_dict=_DEFAULT_DICT_NAME,
        border_fraction: float = 0.15,
        detection_options: dict | None = None,
    ):
        """
        Initialise an icosahedron whose twenty faces are each a ChArUco2 board.

        :param length: Edge length (mm) -- the printed edge length of one of
            the icosahedron's twenty triangular faces, border included.
            Suggested: 50-300.
        :param n_points: Squares per face -- squares along the bottom edge of
            a face. Only the squares a face actually prints take an id of
            their own, so this is what the dictionary runs out of.
            Suggested: 10-12.
        :param aruco_dict: ArUco dictionary -- the marker alphabet, shared out
            twenty ways so that each face carries markers of its own.
        :param border_fraction: Border fraction -- how much of each face is
            blank margin rather than board. The band of tabs around a face's
            board is drawn in it. Suggested: 0.1-0.2.
        :param detection_options: what the detector is told, by the keys
            :meth:`detector_parameterisation` describes. aruco2's grid-board
            detector takes none, so this is always empty.
        """
        super().__init__(inputs=locals(), backend=ARUCO2_BACKEND)

        if isinstance(n_points, bool) or not float(n_points).is_integer():
            raise ValueError(
                f"A CIco2 face is a whole number of squares; got "
                f"n_points={n_points!r}.")
        n_points = int(n_points)
        if n_points < _MIN_POINTS:
            raise ValueError(
                f"A CIco2 face must be at least {_MIN_POINTS} squares along "
                f"its edge; got n_points={n_points}.")
        if length <= 0:
            raise ValueError(
                f"An icosahedron is printed, so it has an edge length; got "
                f"length={length}.")
        if not 0 < border_fraction < 1:
            raise ValueError(
                f"A face is part margin and part board, so its border is "
                f"neither all of it nor none; got "
                f"border_fraction={border_fraction}.")

        # Where this face draws its lattice.  A size past the end of the
        # table cannot be afforded by any dictionary the backend offers, and
        # is left flush so that it fails on the alphabet, below, and says so.
        scale, phase = FACE_LATTICE.get(n_points, (float(n_points), 0.0))

        # A band is a quarter of a square deep and lies in the margin, so the
        # margin has to be at least that; past it a tab runs off the face and
        # round the fold onto its neighbour.  It is the lattice that sets the
        # square and not n_points, and the lattice is the coarser of the two,
        # so the margin a face needs is a little wider than the name suggests.
        min_border = 1 / (2 * scale + 1)
        if border_fraction < min_border:
            quoted = math.ceil(min_border * 1e4 - 1e-9) / 1e4
            raise ValueError(
                f"A {n_points}-square CIco2 face draws a band of tabs a "
                f"quarter of a square deep around its board, in the border, "
                f"so its border fraction must be at least {quoted:.4f}; got "
                f"border_fraction={border_fraction}.")

        self.basis = make_icosahedral()
        self.n_faces = self.basis.n_faces
        self.n_points = n_points
        self.border_fraction = float(border_fraction)
        self._aruco_dict_int = dictionary_id(aruco_dict, ARUCO2_BACKEND)
        self.aruco_dict = aruco_dict

        #: How many squares actually span the board's bottom edge, which is
        #: a little short of :attr:`n_points`.  See :data:`FACE_LATTICE`.
        self.lattice_scale = scale
        #: How far along that edge the squares start, in squares.
        self.lattice_phase = phase

        self.length = length / 1000
        self.board_edge = self.length * (1 - self.border_fraction)
        self.square_size = self.board_edge / self.lattice_scale

        self.cells = clip_lattice_to_face(
            self.basis.base_face, self.lattice_scale,
            (self.lattice_phase, 0.0))
        self.live_corners = corners_touching_cells(self.cells)

        self.board_columns = self.n_points
        self.board_rows = int(self.cells[:, 1].max()) + 1
        self.grid_size = (self.board_columns, self.board_rows)

        # Every square of the rectangle needs an id, printed or not, because
        # the board that reads a face is the rectangle.  Giving each face a
        # whole rectangle's worth would spend more than half the alphabet on
        # squares the triangle cut away -- 22 of every 40 at eight squares to
        # an edge.  Only the squares that are printed get an id of their own;
        # the rest share one block of filler between all twenty faces.  They
        # can be shared because they are never printed, so no image can
        # contain one, and a board still has a distinct id per square, which
        # is what aruco2 requires (a single repeated filler is refused).
        #
        # It buys most of the target's resolution: twelve squares to an edge
        # needs 1920 markers a face at a time and 970 shared, so it fits a
        # thousand-marker dictionary where it did not before.
        printed = sorted(self._printed_cells(), key=lambda cell: cell[::-1])
        self.markers_per_face = len(printed)
        filler_count = self.board_columns * self.board_rows - len(printed)
        needed = self.n_faces * self.markers_per_face + filler_count
        held = dictionary_marker_count(self._aruco_dict_int)
        if needed > held:
            raise ValueError(
                f"A {n_points}-square CIco2 needs {needed} markers -- twenty "
                f"faces of {self.markers_per_face} printed squares, and "
                f"{filler_count} shared between the squares the triangle cut "
                f"away -- and {aruco_dict!r} holds {held}. Use fewer squares "
                f"per face, or a larger dictionary.")

        #: The ids given to squares that are never printed.  One block, shared
        #: by every face, and detectable on none of them.
        self.filler_ids = list(range(
            self.n_faces * self.markers_per_face, needed))
        #: face k's marker ids, row-major over the whole rectangle: its own
        #: block on the squares it prints, and the shared filler on the rest.
        self.face_ids = [
            self._ids_for_face(k, printed) for k in range(self.n_faces)]
        for k, ids in enumerate(self.face_ids):
            face_name = f"Face {k} of a {n_points}-square CIco2"
            refuse_rotation_ambiguous_markers(
                self._aruco_dict_int, ids, aruco_dict, face_name)
            warn_known_false_detections(
                self._aruco_dict_int, ids, aruco_dict, face_name)

        # Where lattice corner (0, 0) sits: the board's triangle is centred
        # in the face, and the lattice starts a fraction of a square along it.
        # Everything drawn -- the squares, the corners, the band -- is placed
        # from here, so the phase is applied once, here, and nowhere else.
        inset = self.length - self.board_edge
        self.board_offset = np.array(
            [inset / 2 + self.lattice_phase * self.square_size,
             inset * TRIANGLE_HEIGHT / 3])

        self._face_rects = [self._build_face_rectangles(k)
                            for k in range(self.n_faces)]

        self.base_face = np.concatenate([
            self.basis.base_face[:, :2] * self.length,
            np.zeros((self.basis.n_corners, 1)),
        ], axis=1)

        self.faceData = FaceToShape(
            face_local_coords=self._face_points(),
            face_transforms=self.basis.face_matrices(),
            scale_factor=self.length,
        )
        self.point_data = self.faceData.point_data
        self._process_data()

    # -- geometry -------------------------------------------------------------

    @property
    def points_per_face(self) -> int:
        """How many corners one face offers."""
        return len(self.live_corners)

    def _ids_for_face(self, face_index: int,
                      printed: list[tuple[int, int]]) -> list[int]:
        """
        Return one face's marker ids, row-major over the whole rectangle.

        The squares this face prints get its own block, in the order they are
        laid out; every other square gets one of the shared filler ids, which
        no face prints and so no image can show.

        :param face_index: which face
        :param printed: the printed cells, in row-major order
        """
        columns = self.board_columns
        ids: list[int | None] = [None] * (columns * self.board_rows)
        for offset, (column, row) in enumerate(printed):
            ids[row * columns + column] = (
                face_index * self.markers_per_face + offset)
        spare = iter(self.filler_ids)
        return [next(spare) if marker is None else marker for marker in ids]

    def _corner_index(self, column: int, row: int) -> int:
        """The gid aruco2 reports a lattice corner by."""
        return int(row) * (self.board_columns + 1) + int(column)

    def _live_corner_lookup(self) -> dict[int, int]:
        """Map a board's corner gid to this face's point index."""
        return {self._corner_index(column, row): index
                for index, (column, row) in enumerate(self.live_corners)}

    def _face_points(self) -> np.ndarray:
        """Return the corners of every face, in face-local metres."""
        all_corners = grid_board_corners(
            self.grid_size, self.square_size,
            origin=(float(self.board_offset[0]), float(self.board_offset[1])))
        keep = [self._corner_index(c, r) for c, r in self.live_corners]
        local = np.asarray(all_corners)[keep]
        points = np.concatenate(
            [local, np.zeros((len(local), 1))], axis=1)
        return np.tile(points[None, ...], (self.n_faces, 1, 1))

    # -- drawing --------------------------------------------------------------

    def _printed_cells(self) -> set[tuple[int, int]]:
        return {(int(c), int(r)) for c, r in self.cells}

    def _build_face_rectangles(self, face_index: int) -> np.ndarray:
        """
        Return the black rectangles one clipped face prints, in local metres.

        The whole rectangular board is laid out first, by the one layout
        ChArUco2 and Ccube2 print from, and then cut to the squares that fell
        inside the triangle -- a run of black cells that crosses the boundary
        is split rather than dropped.  The rectangle's own band is cut away
        with everything else outside, and a band for the clipped shape is put
        back by :meth:`_staircase_band`.
        """
        bits = grid_board_marker_bits(
            self.grid_size, self._aruco_dict_int, ids=self.face_ids[face_index])
        full = grid_board_rectangles(
            self.grid_size, self.square_size, bits,
            origin=(float(self.board_offset[0]), float(self.board_offset[1])))

        square = self.square_size
        ox, oy = float(self.board_offset[0]), float(self.board_offset[1])
        kept: list[tuple[float, float, float, float]] = []
        for x0, y0, x1, y1 in full:
            for column, row in self._printed_cells():
                cx0, cy0 = ox + column * square, oy + row * square
                cx1, cy1 = cx0 + square, cy0 + square
                ix0, iy0 = max(x0, cx0), max(y0, cy0)
                ix1, iy1 = min(x1, cx1), min(y1, cy1)
                if ix1 - ix0 > 1e-12 and iy1 - iy0 > 1e-12:
                    kept.append((ix0, iy0, ix1, iy1))
        kept.extend(self._staircase_band())
        return np.asarray(kept, dtype=np.float64).reshape(-1, 4)

    def _staircase_band(self) -> list[tuple[float, float, float, float]]:
        """
        Return the band of tabs around the clipped board.

        A rectangular board gets a ring of alternating tabs, which is what
        gives its outer corners a black side and a white one.  Clipped, its
        outer boundary is a staircase, and the ring has to follow it.

        The rule is the chessboard's own, rather than anything about edges:
        a square just outside the board that the chessboard would have printed
        *black* is printed, cut back to the band's depth.  ``grid_board_cells``
        puts the inverted marker on squares with an odd ``column + row``, so
        the black ones are the even ones.

        Taking it from the chessboard rather than from the boundary is what
        makes it right on a staircase.  Across an edge the parity flips, so
        "the square outside is black" and "the square inside is white" agree --
        which is why a rule about the inside squares works on a rectangle.
        Across a *corner* the parity does not flip, and a staircase is mostly
        corners, so the two rules disagree there and only this one continues
        the pattern.

        On an unclipped rectangle this draws the real band exactly, less the
        two corner squares aruco2 puts at the far corners of the board, which
        sit on white and are that design's own anchors rather than part of the
        chessboard.
        """
        printed = self._printed_cells()
        square = self.square_size
        depth = band_depth(square)
        ox, oy = float(self.board_offset[0]), float(self.board_offset[1])

        outside = {(column + dx, row + dy)
                   for column, row in printed
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1)} - printed
        tabs: list[tuple[float, float, float, float]] = []
        for column, row in sorted(outside):
            if (column + row) % 2 != 0:
                continue
            x0, y0 = ox + column * square, oy + row * square
            x1, y1 = x0 + square, y0 + square
            # The part of this square nearest the board it borders: a strip
            # along each edge it shares with a printed square...
            if (column, row - 1) in printed:
                tabs.append((x0, y0, x1, y0 + depth))
            if (column, row + 1) in printed:
                tabs.append((x0, y1 - depth, x1, y1))
            if (column - 1, row) in printed:
                tabs.append((x0, y0, x0 + depth, y1))
            if (column + 1, row) in printed:
                tabs.append((x1 - depth, y0, x1, y1))
            # ...and a square in each corner that a printed square only
            # touches diagonally, which a staircase has a great many of.
            for dx, cx in ((-1, x0), (1, x1 - depth)):
                for dy, cy in ((-1, y0), (1, y1 - depth)):
                    if (column + dx, row + dy) in printed:
                        tabs.append((cx, cy, cx + depth, cy + depth))
        return tabs

    def face_rectangles(self, face_index: int) -> np.ndarray:
        """
        Every black shape face ``face_index`` prints.

        :param face_index: which face
        :return: ``(N, 4)`` ``(x0, y0, x1, y1)`` rectangles in face-local metres
        """
        return self._face_rects[face_index]

    def _face_outline(self) -> np.ndarray:
        return self.basis.base_face[:, :2] * self.length

    def _svg_document(
        self,
        border_width: float = 10.0,
        draw_cut_outline: bool = True,
        draw_face_ids: bool = True,
    ) -> tuple[svgwrite.Drawing, float, float]:
        """Build the printable net as one vector document."""
        black: list[np.ndarray] = []
        outlines: list[np.ndarray] = []
        labels: list[tuple[str, np.ndarray, float]] = []

        for face_index in range(self.n_faces):
            affine = self.basis.net_affine(face_index, self.length)
            for x0, y0, x1, y1 in self.face_rectangles(face_index):
                corners = np.array(
                    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                black.append(_affine(corners, affine))
            outline = _affine(self._face_outline(), affine)
            outlines.append(outline)
            if draw_face_ids:
                labels.append((str(face_index), outline.mean(axis=0),
                               self.length * 0.12))

        points = np.vstack(black + outlines)
        border = float(border_width) * 0.001
        low = points.min(axis=0) - border
        high = points.max(axis=0) + border
        width, height = float(high[0] - low[0]), float(high[1] - low[1])

        drawing = svgwrite.Drawing(
            size=(f"{width * 1000.0:.6f}mm", f"{height * 1000.0:.6f}mm"),
            viewBox=f"0 0 {width:.6f} {height:.6f}")
        drawing.add(drawing.rect(
            insert=(0, 0), size=(width, height), fill="white"))

        offset = -low
        for polygon in black:
            drawing.add(drawing.polygon(
                points=[tuple(xy) for xy in polygon + offset],
                fill="black", stroke="none"))
        if draw_cut_outline:
            for outline in outlines:
                drawing.add(drawing.polygon(
                    points=[tuple(xy) for xy in outline + offset],
                    fill="none", stroke="black", stroke_width=0.0002))
        for text, centre, size in labels:
            drawing.add(drawing.text(
                text, insert=tuple(centre + offset), fill="red",
                font_size=f"{size:.6f}", font_family="Arial",
                text_anchor="middle"))
        return drawing, width, height

    def save_to_svg(
        self,
        f_out: Path | str,
        border_width: float = 10.0,
        draw_cut_outline: bool = True,
        draw_face_ids: bool = True,
        suppress_svg_log: bool = False,
    ) -> Path:
        """Save the net as a physically sized vector SVG."""
        f_out = export_path(f_out, ".svg")
        drawing, _, _ = self._svg_document(
            border_width, draw_cut_outline, draw_face_ids)
        with open(f_out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(drawing.tostring())
            handle.flush()
        if (not f_out.exists()) or f_out.stat().st_size == 0:
            raise IOError(f"SVG write failed: {f_out}")
        if not suppress_svg_log:
            logger.info("Saved CIco2 SVG: %s", f_out)
        return f_out

    def save_to_pdf(
        self,
        f_out: Path | str,
        data_format: str = "raster",
        dpi: int = 300,
        border_width: float = 10.0,
        draw_cut_outline: bool = True,
        draw_face_ids: bool = True,
    ) -> Path:
        """Save the net as a raster or vector PDF."""
        from pyCamSet.calibration_targets.cico import _require_cairo
        f_out = export_path(f_out, ".pdf")
        drawing, width, height = self._svg_document(
            border_width, draw_cut_outline, draw_face_ids)
        svg_bytes = drawing.tostring().encode("utf-8")
        cairosvg = _require_cairo()
        if data_format == "vector":
            cairosvg.svg2pdf(bytestring=svg_bytes, write_to=str(f_out))
            logger.info("Saved CIco2 Vector PDF: %s", f_out)
            return f_out
        if data_format != "raster":
            raise ValueError("data_format must be one of: raster, vector")
        from io import BytesIO

        from PIL import Image
        png = cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=max(1, int(round(width * 1000.0 / 25.4 * dpi))),
            output_height=max(1, int(round(height * 1000.0 / 25.4 * dpi))))
        with Image.open(BytesIO(png)) as image:
            image.convert("RGB").save(f_out, resolution=float(dpi))
        logger.info("Saved CIco2 Raster PDF: %s", f_out)
        return f_out

    def to_stl(self, f_out: Path | str) -> Path:
        """
        Write the bare solid as an STL, to print a core to mount faces on.

        :param f_out: where to write it
        """
        return self.basis.to_stl(f_out, edge_length=self.length * 1000.0)

    def _face_texture(self, face_index: int, resolution: int = 512) -> np.ndarray:
        """Rasterise one face, for the 3-D view."""
        from pyCamSet.calibration_targets.cico import _require_cairo
        cairosvg = _require_cairo()
        extent = self.length
        height = extent * TRIANGLE_HEIGHT
        drawing = svgwrite.Drawing(
            size=(f"{extent:.6f}", f"{height:.6f}"),
            viewBox=f"0 0 {extent:.6f} {height:.6f}")
        drawing.add(drawing.rect(
            insert=(0, 0), size=(extent, height), fill="white"))
        for x0, y0, x1, y1 in self.face_rectangles(face_index):
            drawing.add(drawing.rect(
                insert=(x0, y0), size=(x1 - x0, y1 - y0),
                fill="black", stroke="none"))
        from io import BytesIO

        from PIL import Image
        png = cairosvg.svg2png(
            bytestring=drawing.tostring().encode("utf-8"),
            output_width=resolution,
            output_height=max(1, int(resolution * TRIANGLE_HEIGHT)))
        with Image.open(BytesIO(png)) as image:
            return np.asarray(image.convert("RGB"))

    def plot(self, return_scene: bool = False, draw_res: int = 512):
        """
        Show the target as a solid, its faces textured.

        :param return_scene: hand back the pyvista scene rather than showing it
        :param draw_res: how wide each face's texture is rasterised
        """
        textures = [self._face_texture(face, draw_res)
                    for face in range(self.n_faces)]
        scene = self.faceData.draw_meshes(
            self.base_face, textures, return_scene=return_scene)
        if return_scene:
            return scene
        return None

    # -- detection ------------------------------------------------------------

    def find_in_image(
        self,
        image,
        draw: bool = False,
        camera: Camera | None = None,
        wait_len: int = 1,
    ) -> ImageDetection:
        """
        Find the target's corners in an image, one face at a time.

        :param image: the image, uint8 or an integral 0..255 float
        :param draw: show what was found
        :param camera: unused; aruco2's grid-board detector takes no camera
        :param wait_len: how long the drawn window waits; -1 for a keypress
        """
        del camera
        image = _as_uint8_image(image)
        lookup = self._live_corner_lookup()
        keys: list[list[int]] = []
        points: list[np.ndarray] = []

        # aruco2 finds one grid board per call, and a face is only found under
        # its own ids, so each face is looked for on its own.
        for face_index, ids in enumerate(self.face_ids):
            corner_ids, corner_points = detect_grid_board_corners(
                image, self.grid_size, self._aruco_dict_int, self.square_size,
                ids=ids)
            if corner_ids is None:
                continue
            for gid, point in zip(np.asarray(corner_ids).reshape(-1),
                                  np.asarray(corner_points).reshape(-1, 2)):
                index = lookup.get(int(gid))
                if index is None:
                    # A corner of the rectangle the board was cut from that
                    # this face did not print.
                    continue
                keys.append([face_index, index])
                points.append(point)

        if not keys:
            return ImageDetection()
        image_points = np.asarray(points, dtype=np.float64)
        if draw:
            import cv2
            display = np.asarray(image).copy()
            for x, y in image_points:
                cv2.circle(display, (int(round(x)), int(round(y))),
                           3, (0, 0, 255), 1)
            cv2.imshow("CIco2 detections", display)
            cv2.waitKey(wait_len)
        return ImageDetection(
            np.asarray(keys, dtype=np.int64), image_points)


def _affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Push (n, 2) points through a 3x3 affine."""
    homogeneous = np.c_[points, np.ones((len(points), 1))]
    return (matrix @ homogeneous.T).T[:, :2]
