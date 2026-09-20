"""
An icosahedron whose twenty triangular faces each carry a PuzzleBoard window.

The cube version is
:class:`~pyCamSet.calibration_targets.puzzleboard_cube.PuzzleBoardCube`, and
the arrangement is the same one: the printed pattern is a window of the
periodic code field, a different window per face, so a decoded position says
both which face was seen and where on it.  Here the window is then clipped to
the triangle, and only whole squares are printed.

**Why the windows are simply tiled, and not chosen for code distance.**  It
would be reasonable to expect that spacing the twenty windows apart in the
field buys margin against a patch of one face decoding as another.  Measured,
it does not.  With twenty 20x20 windows the smallest Hamming distance between
a 3x3 patch of one face and a 3x3 patch of another is 1, for a row-major
tiling, for a maximally spread layout, and for random disjoint layouts alike:
the base code is a *sub-perfect map*, which is built to make every patch
unique and says nothing about how far apart they are, and no choice of window
can add a property the code does not have.  What placement does decide is
whether the windows *overlap*: two faces sharing any of the field carry
literally identical patches, a distance of 0, which is the one failure worth
designing against.  So the windows are tiled disjointly and deterministically,
and robustness is left where PuzzleBoard itself puts it -- in voting across
the many patches a face shows at once, not in the margin of any one of them.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import svgwrite

from pyCamSet.calibration_targets.core import (
    AbstractTarget, FaceToShape, ImageDetection,
)
from pyCamSet.calibration_targets.core.abstract_target import (
    EXPORT_SUFFIXES, export_path,
)
from pyCamSet.calibration_targets.core.parameters import (
    DocumentedParameters,
    Parameterisation,
)
from pyCamSet.calibration_targets.markers.puzzleboard import (
    PUZZLEBOARD_DETECTOR,
    detect_puzzleboard_image,
)
from pyCamSet.calibration_targets.polyhedra import (
    TRIANGLE_HEIGHT,
    clip_lattice_to_face,
    corners_within_cells,
    make_icosahedral,
)
from pyCamSet.calibration_targets.puzzleboard import _CODE_FIELD, _CODE_SIZE
from pyCamSet.cameras import Camera

logger = logging.getLogger(__name__)

#: Freezes the window layout, so a target built by a later version is not
#: quietly a different target under the same name.
CODE_LAYOUT_VERSION = "puzzle-ico-v1"

#: The windows are laid out four rows of five, which is also how the net is
#: arranged; any disjoint layout would do equally well (see the module
#: docstring), so this one is chosen for being easy to read off the field.
FACE_GRID_COLUMNS = 5
FACE_GRID_ROWS = 4

#: The largest window that still lets twenty of them tile the field.
MAX_FACE_SQUARES = _CODE_SIZE // FACE_GRID_COLUMNS

#: The smallest face worth clipping to.  Below this the triangle holds too few
#: whole squares to leave any corner with all four of its squares printed.
_MIN_POINTS = 6

#: How far the code read for a square is shifted from the square's position.
#:
#: Upstream PuzzleBoard centres its squares on the integer code positions, so
#: the corner it reports for code ``(row, col)`` falls at a half-integer.  The
#: squares here run integer to integer instead, which is the convention
#: :func:`clip_lattice_to_face` works in, and puts the corners on the integers.
#: That half-square difference is a whole number of code positions -- 334,
#: which is 2 * 167, the base code's period -- and it is the same number
#: whichever window is printed and whichever way it is placed, measured across
#: window origins spanning the field.  Reading the code this far back makes a
#: printed window decode to the position it was cut from, so nothing
#: downstream has to know about any of this.
#:
#: Pinned by test_puzzleboard_ico_target.py, which renders a window and checks
#: it decodes to its own origin; if a future detector anchors itself
#: differently, that test fails rather than the target quietly mislabelling
#: every face.
_CODE_PHASE = 334


def _code_at(row: int, column: int) -> int:
    """
    Return the code value a square at a code position is printed with.

    :param row: the square's row in the code field
    :param column: the square's column in the code field
    """
    return int(_CODE_FIELD[(row - _CODE_PHASE) % _CODE_SIZE,
                           (column - _CODE_PHASE) % _CODE_SIZE])


class PuzzleBoardIco(AbstractTarget):
    """An icosahedron of twenty triangular PuzzleBoard faces."""

    DETECTOR_BACKENDS = {"puzzle_board": PUZZLEBOARD_DETECTOR}

    @classmethod
    def construction_parameters(cls, backend: str | None = None) -> Parameterisation:
        """What decides where an icosahedron's corners are."""
        return DocumentedParameters(cls.__init__, "n_points", "length")

    @classmethod
    def export_parameters(cls) -> Parameterisation:
        """How the net is drawn, which is not what the target is."""
        return DocumentedParameters(
            cls.save_printable, "border_width", "draw_cut_outline",
            "draw_face_ids", "dpi")

    @classmethod
    def printable_name(cls, values: dict, kind: str = "svg") -> str:
        return (f"puzzleboard_ico_{int(values['n_points'])}points_"
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
        raise ValueError(f"A PuzzleBoardIco cannot be written as {kind!r}.")

    def __init__(
        self,
        length: float = 100.0,
        n_points: int = 16,
        detection_options: dict | None = None,
    ):
        """
        Initialise an icosahedron of twenty PuzzleBoard faces.

        :param length: Edge length (mm) -- the printed edge length of one of
            the icosahedron's twenty triangular faces. Suggested: 50-300.
        :param n_points: Squares per face -- code squares along the bottom
            edge of a face. The window is clipped to the triangle, so a face
            prints fewer squares than this counts. Suggested: 12-24.
        :param detection_options: what the detector is told, by the keys
            :meth:`detector_parameterisation` describes.
        """
        super().__init__(inputs=locals())

        if n_points < _MIN_POINTS:
            raise ValueError(
                f"A PuzzleBoardIco face must be at least {_MIN_POINTS} squares "
                f"along its edge; got n_points={n_points}. Below that, "
                f"clipping the window to the triangle leaves no corner with "
                f"all four of its squares printed.")
        if n_points > MAX_FACE_SQUARES:
            raise ValueError(
                f"n_points must not exceed {MAX_FACE_SQUARES} for "
                f"{CODE_LAYOUT_VERSION}; five windows across must fit inside "
                f"the {_CODE_SIZE}x{_CODE_SIZE} code field.")
        if length <= 0:
            raise ValueError(
                f"An icosahedron is printed, so it has an edge length; got "
                f"length={length}.")

        self.basis = make_icosahedral()
        self.n_faces = self.basis.n_faces
        self.n_points = int(n_points)
        self.layout_version = CODE_LAYOUT_VERSION
        self.min_width = self.detection_options["min_width"]

        self.length = length / 1000
        self.square_size = self.length / self.n_points

        self.cells = clip_lattice_to_face(self.basis.base_face, self.n_points)
        self.live_corners = corners_within_cells(self.cells)
        if not len(self.live_corners):
            raise ValueError(
                f"A {self.n_points}-square face clips to {len(self.cells)} "
                f"whole squares, which leave no corner with all four of its "
                f"squares printed.")

        self.face_origins = self.face_origins_for_size(self.n_points)

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

    # -- the windows ----------------------------------------------------------

    @classmethod
    def face_origins_for_size(cls, n_points: int) -> tuple[tuple[int, int], ...]:
        """
        Return the twenty windows' origins in the code field, as ``(x, y)``.

        Tiled, so that no two windows share any of the field.  Two windows
        that overlapped would print identical code on two faces, and a patch
        in the shared part would decode to both -- the one placement mistake
        that actually costs anything (see the module docstring).

        :param n_points: the window's size in squares
        :raises ValueError: for a window too large for twenty to fit
        """
        size = int(n_points)
        if size < 2:
            raise ValueError("n_points must be at least 2.")
        if size > MAX_FACE_SQUARES:
            raise ValueError(
                f"n_points must not exceed {MAX_FACE_SQUARES} for "
                f"{CODE_LAYOUT_VERSION}.")
        origins = tuple(
            (column * size, row * size)
            for row in range(FACE_GRID_ROWS)
            for column in range(FACE_GRID_COLUMNS))
        if any(x + size > _CODE_SIZE or y + size > _CODE_SIZE
               for x, y in origins):
            raise ValueError(
                "The window layout runs past the end of the code field.")
        return origins

    @property
    def points_per_face(self) -> int:
        """How many corners one face offers."""
        return len(self.live_corners)

    def _face_points(self) -> np.ndarray:
        """
        Return the corners of every face, in face-local metres.

        Every face carries the same lattice -- the faces are one triangle,
        twenty times -- and differs only in which window of the code is
        printed on it.
        """
        local = self.live_corners * self.square_size
        points = np.concatenate([local, np.zeros((len(local), 1))], axis=1)
        return np.tile(points[None, ...], (self.n_faces, 1, 1))

    def _live_corner_lookup(self) -> dict[tuple[int, int], int]:
        """Map a lattice corner to this face's point index."""
        return {(int(column), int(row)): index
                for index, (column, row) in enumerate(self.live_corners)}

    # -- drawing --------------------------------------------------------------

    def _face_shapes(self, face_index: int):
        """
        Return one face's black squares and its code circles, in local metres.

        A circle sits on the edge between two squares, so it is printed only
        where both of those squares were.  A circle with nothing on one side
        of it is a mark in the margin, not a bit.

        :param face_index: which face
        :return: the black square polygons, and ``(centre, is_white)`` circles
        """
        origin_x, origin_y = self.face_origins[face_index]
        printed = {(int(c), int(r)) for c, r in self.cells}
        square = self.square_size

        polygons = []
        for column, row in sorted(printed):
            if (column + row + origin_x + origin_y) % 2 != 0:
                continue
            x0, y0 = column * square, row * square
            polygons.append(np.array([
                [x0, y0], [x0 + square, y0],
                [x0 + square, y0 + square], [x0, y0 + square]]))

        circles = []
        for column, row in sorted(printed):
            value = _code_at(origin_y + row, origin_x + column)
            # Which bit goes on which edge is PuzzleBoard's, and is easy to
            # get backwards.  Upstream centres its squares on the integer code
            # positions, so its bit 2 sits at (integer, half-integer) -- the
            # middle of a *horizontal* edge -- and its bit 1 at (half-integer,
            # integer), a vertical one.  Here the squares span integer to
            # integer instead, which moves both by half a square: the
            # horizontal edge below this square is at (column + 0.5, row + 1).
            if (column, row + 1) in printed:
                circles.append((
                    np.array([(column + 0.5) * square, (row + 1) * square]),
                    bool(value & 2)))
            if (column + 1, row) in printed:
                circles.append((
                    np.array([(column + 1) * square, (row + 0.5) * square]),
                    bool(value & 1)))
        return polygons, circles

    def _face_outline(self) -> np.ndarray:
        return self.basis.base_face[:, :2] * self.length

    def _svg_document(
        self,
        border_width: float = 10.0,
        draw_cut_outline: bool = True,
        draw_face_ids: bool = True,
    ) -> tuple[svgwrite.Drawing, float, float]:
        """Build the printable net as one vector document."""
        radius = self.square_size / 6.0
        squares, circles, outlines, labels = [], [], [], []

        for face_index in range(self.n_faces):
            affine = self.basis.net_affine(face_index, self.length)
            polygons, face_circles = self._face_shapes(face_index)
            for polygon in polygons:
                squares.append(_affine(polygon, affine))
            for centre, is_white in face_circles:
                circles.append((_affine(centre[None], affine)[0], is_white))
            outline = _affine(self._face_outline(), affine)
            outlines.append(outline)
            if draw_face_ids:
                labels.append((str(face_index), outline.mean(axis=0),
                               self.length * 0.12))

        points = np.vstack(squares + outlines)
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
        for polygon in squares:
            drawing.add(drawing.polygon(
                points=[tuple(xy) for xy in polygon + offset],
                fill="black", stroke="none"))
        for centre, is_white in circles:
            drawing.add(drawing.circle(
                center=tuple(centre + offset), r=radius,
                fill="white" if is_white else "black", stroke="none"))
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
            logger.info("Saved PuzzleBoardIco SVG: %s", f_out)
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
            logger.info("Saved PuzzleBoardIco Vector PDF: %s", f_out)
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
        logger.info("Saved PuzzleBoardIco Raster PDF: %s", f_out)
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
        polygons, circles = self._face_shapes(face_index)
        for polygon in polygons:
            drawing.add(drawing.polygon(
                points=[tuple(xy) for xy in polygon],
                fill="black", stroke="none"))
        for centre, is_white in circles:
            drawing.add(drawing.circle(
                center=tuple(centre), r=self.square_size / 6.0,
                fill="white" if is_white else "black", stroke="none"))
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
        Find the target's corners in an image.

        The detector decodes an absolute position in the periodic field, so
        which face was seen falls out of which window that position lies in --
        there is nothing to match against twenty boards.

        :param image: the image to look in
        :param draw: show what was found
        :param camera: unused; the PuzzleBoard detector takes no camera
        :param wait_len: how long the drawn window waits; -1 for a keypress
        """
        del camera
        point_ids, point_coords = detect_puzzleboard_image(
            image, min_width=self.min_width)
        if len(point_ids) == 0:
            return ImageDetection()

        positions = np.asarray(point_ids, dtype=np.int64)
        coordinates = np.asarray(point_coords, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                "PuzzleBoard detector positions must have shape (n, 2).")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                "PuzzleBoard detector coordinates must have shape (n, 2).")

        lookup = self._live_corner_lookup()
        size = self.n_points
        keys: list[list[int]] = []
        image_points: list[np.ndarray] = []
        for position, coordinate in zip(positions, coordinates):
            row, column = int(position[0]), int(position[1])
            matches = []
            for face_index, (origin_x, origin_y) in enumerate(self.face_origins):
                local_column = column - origin_x
                local_row = row - origin_y
                if 0 <= local_column < size and 0 <= local_row < size:
                    matches.append((face_index, local_row, local_column))
            if len(matches) != 1:
                # The windows are disjoint, so this is a position outside all
                # of them rather than an ambiguous one.
                continue
            face_index, local_row, local_column = matches[0]
            # The detector reports the corner past a code square, which is the
            # lattice corner one along and one up from that square.
            index = lookup.get((local_column + 1, local_row + 1))
            if index is None:
                # A corner of the window this face did not print.
                continue
            keys.append([face_index, index])
            image_points.append(coordinate[::-1])

        if not keys:
            return ImageDetection()
        image_points_array = np.asarray(image_points, dtype=np.float64)
        if draw:
            display = np.asarray(image).copy()
            for point in image_points_array:
                cv2.circle(display, (int(round(point[0])), int(round(point[1]))),
                           3, (0, 0, 255), 1)
            cv2.imshow("PuzzleBoardIco detections", display)
            cv2.waitKey(wait_len)
        return ImageDetection(
            np.asarray(keys, dtype=np.int64), image_points_array)


def _affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Push (n, 2) points through a 3x3 affine."""
    homogeneous = np.c_[points, np.ones((len(points), 1))]
    return (matrix @ homogeneous.T).T[:, :2]
