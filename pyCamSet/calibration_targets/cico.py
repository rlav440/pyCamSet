"""
An icosahedron whose twenty triangular faces each carry a ChArUco board.

The cube version of this is :class:`~pyCamSet.calibration_targets.ccube.Ccube`,
and the difference is worth stating plainly: a distant camera sees three faces
of a cube and ten faces of an icosahedron, whichever way either is turned.  A
rig sees far more of the target at once, and the orientations the target offers
as it is moved are much more finely spaced.

The solid comes from :mod:`~pyCamSet.calibration_targets.polyhedra`, so this
module is only the pattern on a face.  That pattern is a square ChArUco board
clipped to the triangle: whole cells inside it are printed and the rest are
not.  Nothing downstream needs telling, because a board with cells missing is a
partly hidden board, and reading one of those is what a ChArUco detector
already does.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np
import svgwrite
from cv2 import aruco
from matplotlib import pyplot as plt

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
from pyCamSet.calibration_targets.markers.aruco2 import (
    ARUCO2_DETECTOR,
    detect_markers,
    interpolate_board_corners,
    resolve_dictionary,
)
from pyCamSet.calibration_targets.markers.aruco_opencv import (
    ARUCO_OPENCV_DETECTOR, marker_bit_grid,
)
from pyCamSet.calibration_targets.markers.backend_registry import (
    ARUCO1_BACKEND,
    dict_names_for_backend,
    dictionary_id,
)
from pyCamSet.calibration_targets.markers.legacy_probe import (
    should_warn_legacy_mismatch,
)
from pyCamSet.calibration_targets.polyhedra import (
    TRIANGLE_HEIGHT,
    clip_lattice_to_face,
    corners_within_cells,
    make_icosahedral,
)
from pyCamSet.cameras import Camera
from pyCamSet.utils.general_utils import (
    downsample_valid, split_aruco_dictionary,
)

logger = logging.getLogger(__name__)

#: The smallest face worth clipping a board to.  Below six cells to an edge a
#: triangle holds too few whole cells to leave any corner with all four of its
#: cells printed, and a face with no corners is not a calibration target.
_MIN_POINTS = 6

#: The dictionary an icosahedron is printed with unless another is asked for.
#: It is split twenty ways, so it needs markers to spare.
_DEFAULT_DICT_NAME = "DICT_4X4_1000"

#: aruco1 and aruco2 disagree on these, so they are not offered as a choice.
_UNOFFERED_DICT_PREFIXES = ("DICT_APRILTAG_",)


class CIco(AbstractTarget):
    """An icosahedron of twenty triangular ChArUco faces."""

    DETECTOR_BACKENDS = {
        "aruco1": ARUCO_OPENCV_DETECTOR,
        "aruco2": ARUCO2_DETECTOR,
    }

    @classmethod
    def construction_parameters(cls, backend: str | None = None) -> Parameterisation:
        """
        What decides where an icosahedron's corners are, and how it prints.

        :param backend: accepted for the signature every target shares, and
            not used: the detector is chosen in the detection phase, and the
            dictionaries offered are the ones both detectors print alike.
        """
        return DocumentedParameters(
            cls.__init__,
            "n_points", "length", "border_fraction", "aruco_dict", "legacy",
            choices={
                "aruco_dict": exclude_by_prefix(
                    dict_names_for_backend(ARUCO1_BACKEND),
                    *_UNOFFERED_DICT_PREFIXES,
                )
            },
        )

    @classmethod
    def export_parameters(cls) -> Parameterisation:
        """How a CIco net is drawn, which is not what it is."""
        return DocumentedParameters(
            cls.save_printable, "border_width", "draw_cut_outline",
            "draw_face_ids", "dpi")

    @classmethod
    def printable_name(cls, values: dict, kind: str = "svg") -> str:
        return (f"cico_{int(values['n_points'])}points_"
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
        :param draw_cut_outline: Draw cut outline -- an outline to cut the
            net out along. Suggested: on.
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
        raise ValueError(f"A CIco cannot be written as {kind!r}.")

    def __init__(
        self,
        length: float = 100.0,
        n_points: int = 10,
        aruco_dict=_DEFAULT_DICT_NAME,
        border_fraction: float = 0.1,
        legacy: bool = False,
        marker_backend: str = "aruco1",
        detection_options: dict | None = None,
    ):
        """
        Initialise an icosahedron whose twenty faces are each a ChArUco board.

        :param length: Edge length (mm) -- the printed edge length of one of
            the icosahedron's twenty triangular faces, border included.
            Suggested: 50-300.
        :param n_points: Squares per face -- chessboard squares along the
            bottom edge of a face. The board is clipped to the triangle, so a
            face carries fewer squares than this counts. Suggested: 8-12.
        :param aruco_dict: ArUco dictionary -- the marker alphabet, split
            twenty ways so that each face carries markers of its own.
        :param border_fraction: Border fraction -- how much of each face is
            blank margin rather than board. Detection: too little and markers
            near an edge are cut by the fold. Suggested: 0.05-0.15.
        :param legacy: Legacy pattern -- which of OpenCV's two marker layouts
            the faces were printed to; must match how the solid was printed,
            since detection never switches patterns automatically. Detection:
            the wrong one finds every marker and no corners. Suggested: off.
        :param marker_backend: the detector the target is read with,
            "aruco1" (OpenCV) or "aruco2" (the aruco2 package). Chosen in the
            detection phase rather than when the target is made: both print
            the offered dictionaries identically. Defaults to "aruco1".
        :param detection_options: what the detector is told, by the keys
            :meth:`detector_parameterisation` describes.
        """
        super().__init__(inputs=locals(), backend=marker_backend)

        if n_points < _MIN_POINTS:
            raise ValueError(
                f"A CIco face must be at least {_MIN_POINTS} squares along its "
                f"edge; got n_points={n_points}. Below that, clipping the "
                f"board to the triangle leaves no corner with all four of its "
                f"squares printed.")
        if length <= 0:
            raise ValueError(
                f"An icosahedron is printed, so it has an edge length; got "
                f"length={length}.")
        if not 0 < border_fraction < 1:
            raise ValueError(
                f"A face is part margin and part board, so its border is "
                f"neither all of it nor none; got "
                f"border_fraction={border_fraction}.")

        self.basis = make_icosahedral()
        self.n_faces = self.basis.n_faces
        self.n_points = int(n_points)
        self.border_fraction = float(border_fraction)
        self.marker_backend = marker_backend
        self.legacy = bool(legacy)

        # A cv2.aruco.Dictionary is also accepted, and is passed through.
        if isinstance(aruco_dict, (int, str)):
            self._aruco_dict_int = dictionary_id(aruco_dict, marker_backend)
            aruco_dict = self._aruco_dict_int
        else:
            self._aruco_dict_int = None
        self.aruco_dict = aruco_dict

        self.length = length / 1000
        #: The edge of the triangle the board is clipped to, inside the margin.
        self.board_edge = self.length * (1 - self.border_fraction)
        self.square_size = self.board_edge / self.n_points

        # Which whole squares fall inside the triangle, and which corners
        # those squares leave with all four of their number printed.  Both are
        # in lattice units, and both are the same on every face: the faces of
        # an icosahedron are one triangle, twenty times.
        self.cells = clip_lattice_to_face(
            self.basis.base_face, self.n_points)
        self.live_corners = corners_within_cells(self.cells)
        if not len(self.live_corners):
            raise ValueError(
                f"A {self.n_points}-square face clips to {len(self.cells)} "
                f"whole squares, which leave no corner with all four of its "
                f"squares printed.")

        #: The rectangle the clipped board is cut out of.
        self.board_columns = self.n_points
        self.board_rows = int(self.cells[:, 1].max()) + 1

        # Twenty faces are cut from one marker alphabet, and it ends.  The
        # whole rectangle's markers are reserved, not just the printed ones:
        # the board object that reads a face is the rectangle, so every one of
        # its markers needs an id, whether or not it was printed.
        #
        # TODO: CIco2 spends far less of the alphabet on the same clipping --
        # only the squares a face prints take an id of their own, and the ones
        # the triangle cut away share one block of filler between all twenty
        # faces (see cico2.CIco2 and its _ids_for_face).  The same would let a
        # CIco face go from twelve squares to an edge to sixteen, and from 29
        # corners to 65, inside an ordinary thousand-marker dictionary.  It was
        # tried and backed out: at sixteen squares only 58-65 of the 65 corners
        # come back, because the ones near the triangle's tip have too few
        # markers around them for ChArUco interpolation, where twelve squares
        # returns all 29 every time.  Worth revisiting with a rule that drops
        # the corners that cannot be interpolated rather than claiming them.
        self.markers_per_face = (self.board_columns * self.board_rows) // 2
        resolved_dict = resolve_dictionary(aruco_dict, marker_backend)
        held = int(resolved_dict.bytesList.shape[0])
        if self.n_faces * self.markers_per_face > held:
            raise ValueError(
                f"A {self.n_points}-square icosahedron needs "
                f"{self.n_faces * self.markers_per_face} markers, twenty faces "
                f"of {self.markers_per_face}, and this dictionary holds "
                f"{held}. Use fewer squares per face, or a larger dictionary.")
        self.a_dicts = split_aruco_dictionary(
            self.markers_per_face, resolved_dict)

        self.boards = [
            aruco.CharucoBoard(
                (self.board_columns, self.board_rows), self.square_size,
                markerLength=0.75 * self.square_size,
                dictionary=a_dict,
            )
            for a_dict in self.a_dicts][:self.n_faces]
        if legacy:
            for board in self.boards:
                board.setLegacyPattern(True)

        # Where the clipped board sits on the face: the board's triangle and
        # the face's triangle share a centroid, so the margin is even.
        inset = self.length - self.board_edge
        self.board_offset = np.array(
            [inset / 2, inset * TRIANGLE_HEIGHT / 3])

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

        self.board_detectors = None
        self.given_legacy_warning = False
        #: The warning ``_warn_legacy_once`` fired, for a worker process to
        #: hand back: its own ``logger.warning`` never reaches the caller.
        self.legacy_warning_message: str | None = None
        #: So that two threads sharing this target cannot both warn.
        self._legacy_warning_lock = threading.Lock()

    # -- geometry -------------------------------------------------------------

    @property
    def points_per_face(self) -> int:
        """How many corners one face offers."""
        return len(self.live_corners)

    def _board_corner_index(self, column: int, row: int) -> int:
        """
        Return the ChArUco corner id at a lattice corner.

        A board's interior corners are numbered row-major, and the corner at
        lattice ``(column, row)`` is the interior corner one square in from
        the board's origin, so the lattice's first interior corner is ``(1,
        1)`` and the board's is ``0``.
        """
        return (column - 1) + (row - 1) * (self.board_columns - 1)

    def _live_corner_lookup(self) -> dict[int, int]:
        """Map a board's corner id to this face's point index."""
        return {
            self._board_corner_index(column, row): index
            for index, (column, row) in enumerate(self.live_corners)
        }

    def _face_points(self) -> np.ndarray:
        """
        Return the corners of every face, in face-local metres.

        Every face carries the same lattice, because every face is the same
        triangle; only the markers printed on it differ.
        """
        local = (self.live_corners * self.square_size) + self.board_offset
        points = np.concatenate(
            [local, np.zeros((len(local), 1))], axis=1)
        return np.tile(points[None, ...], (self.n_faces, 1, 1))

    def _printed_cells(self) -> set[tuple[int, int]]:
        return {(int(column), int(row)) for column, row in self.cells}

    def _face_polygons(self, face_index: int) -> list[np.ndarray]:
        """
        Return the black shapes printed on one face, in face-local metres.

        The chessboard's black squares, and the black cells of every marker
        that fell inside the triangle.  A marker in a square that was clipped
        away is not drawn, and the square it would have sat in is blank.
        """
        printed = self._printed_cells()
        square = self.square_size
        polygons: list[np.ndarray] = []

        for column, row in sorted(printed):
            if (column + row) % 2 != 0:
                continue
            x0, y0 = column * square, row * square
            polygons.append(np.array([
                [x0, y0], [x0 + square, y0],
                [x0 + square, y0 + square], [x0, y0 + square],
            ]) + self.board_offset)

        board = self.boards[face_index]
        dictionary = board.getDictionary()
        ids = np.asarray(board.getIds()).reshape(-1).astype(int)
        for marker_id, corners in zip(ids, board.getObjPoints()):
            quad = np.asarray(corners, dtype=float).reshape(-1, 3)[:, :2]
            centre = quad.mean(axis=0)
            cell = (int(centre[0] // square), int(centre[1] // square))
            if cell not in printed:
                continue
            grid = marker_bit_grid(dictionary, marker_id)
            rows, columns = grid.shape
            for r in range(rows):
                for c in range(columns):
                    if grid[r, c] != 1:
                        continue
                    u0, u1 = c / columns, (c + 1) / columns
                    v0, v1 = r / rows, (r + 1) / rows
                    polygons.append(np.array([
                        _bilinear(quad, u0, v0), _bilinear(quad, u1, v0),
                        _bilinear(quad, u1, v1), _bilinear(quad, u0, v1),
                    ]) + self.board_offset)
        return polygons

    def _face_outline(self) -> np.ndarray:
        """The triangle a face is cut out as, in face-local metres."""
        return self.basis.base_face[:, :2] * self.length

    # -- printing -------------------------------------------------------------

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
            for polygon in self._face_polygons(face_index):
                black.append(_affine(polygon, affine))
            outline = _affine(self._face_outline(), affine)
            outlines.append(outline)
            if draw_face_ids:
                # Near the middle of the face, which for a triangle is the
                # only place that is inside it whichever way it is turned.
                centre = outline.mean(axis=0)
                labels.append((str(face_index), centre, self.length * 0.12))

        points = np.vstack(black + outlines)
        border = float(border_width) * 0.001
        low = points.min(axis=0) - border
        high = points.max(axis=0) + border
        width, height = float(high[0] - low[0]), float(high[1] - low[1])

        drawing = svgwrite.Drawing(
            size=(f"{width * 1000.0:.6f}mm", f"{height * 1000.0:.6f}mm"),
            viewBox=f"0 0 {width:.6f} {height:.6f}",
        )
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
                    fill="none", stroke="black",
                    stroke_width=0.0002))
        for text, centre, size in labels:
            drawing.add(drawing.text(
                text, insert=tuple(centre + offset),
                fill="red", font_size=f"{size:.6f}",
                font_family="Arial", text_anchor="middle"))
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
            logger.info("Saved CIco SVG: %s", f_out)
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
        f_out = export_path(f_out, ".pdf")
        drawing, width, height = self._svg_document(
            border_width, draw_cut_outline, draw_face_ids)
        svg_bytes = drawing.tostring().encode("utf-8")
        cairosvg = _require_cairo()
        if data_format == "vector":
            cairosvg.svg2pdf(bytestring=svg_bytes, write_to=str(f_out))
            logger.info("Saved CIco Vector PDF: %s", f_out)
            return f_out
        if data_format != "raster":
            raise ValueError("data_format must be one of: raster, vector")
        from io import BytesIO

        from PIL import Image
        png = cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=max(1, int(round(width * 1000.0 / 25.4 * dpi))),
            output_height=max(1, int(round(height * 1000.0 / 25.4 * dpi))),
        )
        with Image.open(BytesIO(png)) as image:
            image.convert("RGB").save(f_out, resolution=float(dpi))
        logger.info("Saved CIco Raster PDF: %s", f_out)
        return f_out

    def to_stl(self, f_out: Path | str) -> Path:
        """
        Write the bare solid as an STL, to print a core to mount faces on.

        A twenty-faced net folded by hand is only as accurate as the folding,
        and the geometry a calibration reports is the geometry of the solid,
        not of the paper.  The written solid is built from the same transforms
        the target's own points are, so the two cannot disagree.

        :param f_out: where to write it
        """
        return self.basis.to_stl(f_out, edge_length=self.length * 1000.0)

    def _face_texture(self, face_index: int, resolution: int = 512) -> np.ndarray:
        """Rasterise one face, for the 3-D view."""
        cairosvg = _require_cairo()
        extent = self.length
        height = extent * TRIANGLE_HEIGHT
        drawing = svgwrite.Drawing(
            size=(f"{extent:.6f}", f"{height:.6f}"),
            viewBox=f"0 0 {extent:.6f} {height:.6f}")
        drawing.add(drawing.rect(
            insert=(0, 0), size=(extent, height), fill="white"))
        for polygon in self._face_polygons(face_index):
            drawing.add(drawing.polygon(
                points=[tuple(xy) for xy in polygon],
                fill="black", stroke="none"))
        drawing.add(drawing.text(
            str(face_index), insert=(extent * 0.5, height * 0.9),
            fill="red", font_size=f"{extent * 0.12:.6f}",
            font_family="Arial", text_anchor="middle"))
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

    def _warn_legacy_once(self, face_index: int, board) -> None:
        """
        Warn once per target that a face looks printed to the other pattern.

        Shared wording with Ccube's, plus the face index: detection never
        switches patterns itself, so this is a warning only.

        :param face_index: the face that triggered it
        :param board: that face's board
        """
        with self._legacy_warning_lock:
            if self.given_legacy_warning:
                return
            self.given_legacy_warning = True
            message = (
                f"Face {face_index} of this CIco found markers but no corners, "
                f"which is what a board printed to OpenCV's other ChArUco "
                f"pattern looks like. This target was built with "
                f"legacy={self.legacy}; if the printed solid used the other "
                f"pattern, rebuild it with legacy={not self.legacy}.")
        self.legacy_warning_message = message
        logger.warning(message)

    def find_in_image(
        self,
        image,
        draw: bool = False,
        camera: Camera | None = None,
        wait_len: int = 1,
    ) -> ImageDetection:
        """
        Find the target's corners in an image.

        :param image: the image to look in
        :param draw: show what was found
        :param camera: an optional camera model, for more accurate detections
        :param wait_len: how long the drawn window waits; -1 for a keypress
        """
        del camera
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = _as_uint8(image)

        lookup = self._live_corner_lookup()
        keys: list[list[int]] = []
        points: list[np.ndarray] = []

        if draw:
            preview = image.copy()
            factor = max(
                int(min(np.array(preview.shape[:2]) / [640, 480])), 1)
            preview = downsample_valid(preview, factor).astype(np.uint8)
            if preview.ndim == 2:
                preview = np.tile(preview[..., None], (1, 1, 3))

        if self.marker_backend == "aruco2":
            markers = detect_markers(image, self._aruco_dict_int)
            for face_index, board in enumerate(self.boards):
                face_markers = [
                    (gid - face_index * self.markers_per_face, corners)
                    for gid, corners in markers
                    if gid // self.markers_per_face == face_index
                ]
                if not face_markers:
                    continue
                corner_ids, corner_points = interpolate_board_corners(
                    image, board, face_markers,
                    warn_legacy=(
                        None if self.given_legacy_warning else
                        (lambda _i=face_index, _b=board:
                         self._warn_legacy_once(_i, _b))),
                )
                if corner_ids is None:
                    continue
                _collect(keys, points, face_index, corner_ids, corner_points,
                         lookup)
        else:
            if self.board_detectors is None:
                # Built here rather than in __init__: twenty detectors are
                # expensive, and a target is often made only to be printed.
                self.board_detectors = [
                    ARUCO_OPENCV_DETECTOR.build_detector(
                        board, self.detection_options)
                    for board in self.boards
                ]
            for face_index, detector in enumerate(self.board_detectors):
                corners, corner_ids, marker_points, _ = detector.detectBoard(image)
                if corners is None and marker_points is not None:
                    if (not self.given_legacy_warning):
                        self._warn_legacy_once(face_index, self.boards[face_index])
                if corner_ids is None:
                    continue
                _collect(
                    keys, points, face_index,
                    np.asarray(corner_ids).reshape(-1),
                    np.asarray(corners).reshape(-1, 2), lookup)

        if draw:
            for x, y in points:
                cv2.circle(preview, (int(x / factor), int(y / factor)),
                           3, (0, 0, 255), 1)
            cv2.imshow("detections", preview)
            cv2.waitKey(wait_len)
        if not keys:
            return ImageDetection()
        return ImageDetection(keys=keys, image_points=points)


def _collect(keys, points, face_index, corner_ids, corner_points, lookup):
    """Keep the corners a clipped face actually has, renumbered to its points."""
    for corner_id, point in zip(np.asarray(corner_ids).reshape(-1),
                                np.asarray(corner_points).reshape(-1, 2)):
        index = lookup.get(int(corner_id))
        if index is None:
            # A corner of the rectangle the board was cut from, which this
            # face did not print. Nothing should report one; if something
            # does, it is not a corner of this target.
            continue
        keys.append([face_index, index])
        points.append(point)


def _bilinear(quad: np.ndarray, u: float, v: float) -> np.ndarray:
    """Interpolate inside a quad given as top left, top right, bottom right, bottom left."""
    return ((1 - u) * (1 - v) * quad[0] + u * (1 - v) * quad[1]
            + u * v * quad[2] + (1 - u) * v * quad[3])


def _affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Push (n, 2) points through a 3x3 affine."""
    homogeneous = np.c_[points, np.ones((len(points), 1))]
    return (matrix @ homogeneous.T).T[:, :2]


def _as_uint8(image: np.ndarray) -> np.ndarray:
    """Accept an integral floating-point image, and refuse anything else."""
    convertible = (
        np.issubdtype(image.dtype, np.floating)
        and image.size > 0
        and np.all(np.isfinite(image))
        and np.allclose(image, np.round(image))
        and float(np.min(image)) >= 0.0
        and float(np.max(image)) <= 255.0
    )
    if not convertible:
        raise ValueError(
            "CIco detection requires a uint8 image or an integral "
            f"floating-point image in the range 0..255; got dtype "
            f"{image.dtype}.")
    return image.astype(np.uint8)


def _require_cairo():
    """Return cairosvg, or say what to install."""
    try:
        import cairosvg
    except OSError as error:
        raise OSError(
            f"{error}\n\n"
            "pyCamSet's target-generation code requires the native 'cairo' "
            "library, which cairosvg requires but pip cannot install on its "
            "own.\nInstall the native cairo library for your platform, then "
            "re-import pyCamSet:\n"
            "  - conda (Windows/Linux/macOS):  conda install -c conda-forge cairo\n"
            "  - Debian/Ubuntu:                 apt install libcairo2\n"
            "  - macOS (Homebrew):              brew install cairo\n"
            "  - Windows (no conda):            install GTK/cairo and put the DLL on PATH"
        ) from error
    return cairosvg
