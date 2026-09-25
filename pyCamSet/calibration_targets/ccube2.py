from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import cv2
import numpy as np
import svgwrite
from cv2 import aruco
from PIL import Image

logger = logging.getLogger(__name__)

from pyCamSet.calibration_targets.core import (
    AbstractTarget, ImageDetection, FaceToShape, exclude_by_prefix,
)
from pyCamSet.calibration_targets.core.abstract_target import (
    EXPORT_SUFFIXES, export_path)
from pyCamSet.calibration_targets.core.parameters import (
    DocumentedParameters,
    Parameterisation,
)
from pyCamSet.calibration_targets.markers.aruco2 import _as_uint8_image
from pyCamSet.calibration_targets.markers.backend_registry import (
    ARUCO2_BACKEND,
    dict_names_for_backend,
    dictionary_id,
)
from pyCamSet.calibration_targets.markers.aruco2_gridboard import (
    CHARUCO2_DETECTOR,
    detect_grid_board_corners,
    dictionary_marker_count,
    grid_board_marker_bits,
    refuse_rotation_ambiguous_markers,
    warn_known_false_detections,
)
from pyCamSet.calibration_targets.markers.gridboard_layout import (
    band_depth,
    grid_board_corners,
    grid_board_rectangles,
    rasterise_rectangles,
    rectangles_svg_path,
)
from pyCamSet.calibration_targets.charuco2 import _EXCLUDED_DICT_PREFIXES
# The cube itself is Ccube's: the same face transforms, the same net and the
# same way of placing a face in it. Only what is printed on a face differs.
from pyCamSet.calibration_targets.ccube import (
    NET_FORMS,
    TFORMS,
    Ccube,
)
from pyCamSet.cameras import Camera
from pyCamSet.utils.general_utils import make_4x4h_tform, downsample_valid

#: The smallest face a cube can have. A ChArUco2 board needs at least two
#: markers each way to be a board (see ChArUco2's own minimum), and aruco2
#: keeps only a connected group of two or more markers per detection.
_MIN_POINTS = 2

#: The dictionary a cube is printed with unless another is asked for. Every
#: square of every face carries its own marker, so it needs six faces' worth.
_DEFAULT_DICT_NAME = "DICT_4X4_1000"

#: A face number's height, as a fraction of the band it is drawn in -- small
#: enough to leave a quarter of the band white above and below it.
_LABEL_BAND_FRACTION = 0.5

#: How tall a digit is in Arial, as a fraction of its font size, so an SVG
#: face number is sized by the height it prints at rather than its em box.
_ARIAL_DIGIT_HEIGHT = 0.716


class Ccube2(AbstractTarget):
    """
    A cube whose six faces are each a ChArUco2 board.

    Ccube's cube, net and face transforms, with every face printed as
    aruco2's ``GridBoard`` design (see
    :class:`~pyCamSet.calibration_targets.charuco2.ChArUco2`): a marker
    on every square, and ``(n_points+1)^2`` corners per face, the face's own
    border included.

    Each face is its own board with its own marker ids -- face ``k`` uses
    ``k * n * n .. (k+1) * n * n - 1``, row-major -- so a face is detected
    independently of the others and a corner is keyed ``[face, gid]`` with
    ``gid = row * (n_points+1) + col``. aruco2's grid-board detector finds
    one board per call, so a cube is six calls per image.

    **Printing is true vector.** Every face is drawn from the one layout in
    :mod:`pyCamSet.calibration_targets.markers.gridboard_layout`, the one
    ChArUco2 prints from, and a face's tabbed band lies in its blank margin.

    **Ccube2 has not been validated on a real printed and photographed
    cube.** Every check behind it runs against rendered images, including a
    synthetic projection of the cube, not against a camera photograph of a
    folded one. Treat detection quality on a real capture as unverified until
    it has been.
    """

    DETECTOR_BACKENDS = {ARUCO2_BACKEND: CHARUCO2_DETECTOR}

    @classmethod
    def construction_parameters(cls, backend: str | None = None) -> Parameterisation:
        """What decides where a cube's corners are, and how it prints."""
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
        """How a Ccube2 net is drawn, which is not what it is."""
        return DocumentedParameters(
            cls.save_printable, "border_width", "draw_cut_outline",
            "draw_board_ids", "individual_faces")

    @classmethod
    def printable_name(cls, values: dict, kind: str = "svg") -> str:
        return (f"ccube2_{int(values['n_points'])}points_"
                f"{float(values['length']):g}mm{EXPORT_SUFFIXES[kind]}")

    def save_printable(self, path, kind: str = "svg", border_width: float = 10.0,
                       draw_cut_outline: bool = True, draw_board_ids: bool = True,
                       individual_faces: bool = False) -> Path:
        """
        Write this cube as a net to print.

        :param path: where to write it
        :param kind: one of :data:`EXPORT_KINDS`
        :param border_width: Net border (mm) -- the margin drawn around the
            folded net. Suggested: 10.
        :param draw_cut_outline: Draw cut outline -- an outline to cut the
            net out along. Suggested: on.
        :param draw_board_ids: Draw face numbers -- a number on each face,
            for folding it the right way up. Suggested: on.
        :param individual_faces: One face per page -- print each face
            separately rather than as one net. For a cube too large to fit
            a page. PDF only, and the pages are raster even for a vector
            PDF. Suggested: off.
        :raises ValueError: for a format a cube cannot be written as
        """
        if kind == "svg":
            return self.save_to_svg(
                path, border_width=border_width,
                draw_cut_outline=draw_cut_outline, draw_board_ids=draw_board_ids)
        if kind in ("pdf_vector", "pdf_raster"):
            return self.save_to_pdf(
                path, border_width=border_width,
                draw_cut_outline=draw_cut_outline, draw_board_ids=draw_board_ids,
                individual_faces=individual_faces,
                data_format="vector" if kind == "pdf_vector" else "raster")
        raise ValueError(f"A Ccube2 cannot be written as {kind!r}.")

    def __init__(self, length: float = 20.0, n_points: int = 5,
                 aruco_dict=_DEFAULT_DICT_NAME,
                 draw_res=(1000, 1000),
                 border_fraction: float = 0.1,
                 line_fraction: float = 0.003,
                 detection_options: dict | None = None,
                 ):
        """
        Initialises a cube whose six faces are each a ChArUco2 board.

        :param length: Cube edge (mm) -- the printed edge length of the
            cube, in millimetres, border included. Suggested: 20-200.
        :param n_points: Squares per face -- marker squares along one edge
            of one of the cube's six faces. Fewer than 5 needs a border
            fraction above the default 0.1. Suggested: 5-8.
        :param aruco_dict: ArUco dictionary -- the marker alphabet. Every
            square of every face carries a marker of its own, so it must
            hold six faces' worth.
        :param draw_res: the resolution each face texture is drawn at.
        :param border_fraction: Border fraction -- how much of each face is
            blank margin rather than board. The band of tabs around each
            face's board is drawn in it, so it must be at least
            1 / (2 * squares per face + 1). Suggested: 0.1-0.2.
        :param line_fraction: the thickness of a face's edge line, as a
            fraction of the face width.
        :param detection_options: what the detector is told, by the keys
            :meth:`detector_parameterisation` describes. aruco2's grid-board
            detector takes none, so this is always empty for Ccube2.
        """
        super().__init__(inputs=locals(), backend=ARUCO2_BACKEND)

        # A face is a whole number of squares.  Checked first, so every check
        # below is of the face actually built, not of a value truncated later.
        if isinstance(n_points, bool) or not float(n_points).is_integer():
            raise ValueError(
                f"A Ccube2 face is a whole number of squares; got "
                f"n_points={n_points!r}.")
        n_points = int(n_points)
        if n_points < _MIN_POINTS:
            raise ValueError(
                f"A Ccube2 face must be at least {_MIN_POINTS}x{_MIN_POINTS} "
                f"squares; got n_points={n_points}.")
        if length <= 0:
            raise ValueError(
                f"A cube is printed, so it has an edge length; got "
                f"length={length}.")
        if not 0 < border_fraction < 1:
            raise ValueError(
                f"A face is part margin and part board, so its border is "
                f"neither all of it nor none; got "
                f"border_fraction={border_fraction}.")
        # A face's band is a quarter of a square deep and lies in the margin,
        # half the border on each side: s/4 <= L*bf/2 with s = L(1-bf)/n is
        # bf >= 1/(2n+1).  Past that, a tab would run off the face and round
        # the fold onto its neighbour.
        min_border = 1 / (2 * n_points + 1)
        if border_fraction < min_border:
            # Rounded up, so the number quoted is one this check accepts.
            quoted = math.ceil(min_border * 1e4 - 1e-9) / 1e4
            raise ValueError(
                f"A {n_points}x{n_points} Ccube2 face draws a band of tabs a "
                f"quarter of a square deep around its board, in the border, "
                f"so its border fraction must be at least 1/{2 * n_points + 1} "
                f"= {quoted:.4f}; got border_fraction={border_fraction}.")

        self.n_points = n_points
        self.input_border_fraction = border_fraction
        self.line_fraction = line_fraction
        self._aruco_dict_int = dictionary_id(aruco_dict, ARUCO2_BACKEND)

        n_squares = self.n_points ** 2
        self.markers_per_face = n_squares
        if 6 * n_squares > (held := dictionary_marker_count(self._aruco_dict_int)):
            # The only ceiling a cube has: every square of six faces is cut
            # from one marker alphabet, and it ends.
            raise ValueError(
                f"A {n_points}x{n_points} Ccube2 needs {6 * n_squares} markers, "
                f"six faces of {n_squares}, and {aruco_dict!r} holds {held}.")

        self.length = length / 1000
        self.square_size = self.length * (1 - border_fraction) / self.n_points
        #: where each face's board starts, from the face's own corner, in m
        self.margin = self.length * border_fraction / 2
        self.grid_size = (self.n_points, self.n_points)
        #: face k's marker ids, row-major -- disjoint ranges, one per face
        self.face_ids = [
            list(range(k * n_squares, (k + 1) * n_squares)) for k in range(6)]
        for k, ids in enumerate(self.face_ids):
            face_name = f"Face {k} of a {n_points}x{n_points} Ccube2"
            refuse_rotation_ambiguous_markers(
                self._aruco_dict_int, ids, aruco_dict, face_name)
            warn_known_false_detections(
                self._aruco_dict_int, ids, aruco_dict, face_name)

        self.draw_res = draw_res
        self.dpi = self.draw_res[0] / self.length / 39.3701  # inch conversion
        self._face_rects = [
            grid_board_rectangles(
                self.grid_size, self.square_size,
                grid_board_marker_bits(self.grid_size, self._aruco_dict_int, ids=ids),
                origin=(self.margin, self.margin))
            for ids in self.face_ids
        ]
        self.textures = [self.face_texture(k) for k in range(6)]

        # (n+1)^2 corners per face, row-major, so a corner's index is the
        # gid aruco2 reports it by; the same lattice on every face, placed
        # on the cube by Ccube's transforms.
        corners = grid_board_corners(
            self.grid_size, self.square_size, origin=(self.margin, self.margin))
        board_coords = np.concatenate(
            [corners, np.zeros((corners.shape[0], 1))], axis=-1)
        self.base_face = np.array([
                            [0, self.length, 0],
                            [self.length, self.length, 0],
                            [self.length, 0, 0],
                            [0, 0, 0],
                        ])
        self.faceData = FaceToShape(
            face_local_coords=np.tile(board_coords[None], (6, 1, 1)),
            face_transforms=[make_4x4h_tform(*t) for t in TFORMS],
            scale_factor=self.length,
        )
        self.point_data = self.faceData.point_data
        self._process_data()

    # -- a face ---------------------------------------------------------------

    def face_rectangles(self, face_index: int) -> np.ndarray:
        """
        Every black shape face ``face_index``'s board prints, band included.

        :return: ``(N, 4)`` ``(x0, y0, x1, y1)`` rectangles in metres, in
            face-local coordinates -- the face's top-left corner at the origin,
            x to the right and y down, the same frame :attr:`point_data` is
            placed on the cube from.
        """
        return self._face_rects[face_index]

    def face_label_anchor(self) -> tuple[float, float, float]:
        """
        Where a face number goes, in face-local metres.

        In the white part of the band below the board, under the standard
        (black-bordered) bottom-row square nearest the middle: that stretch
        of band has no tab, and a digit half the band tall, centred in it,
        is a quarter of the band clear of the board above and of the face's
        edge below, so it cannot touch a marker, a tab or a corner square.

        :return: ``(x_centre, y_centre, height)`` of the number.
        """
        n = self.n_points
        # Square (x, n-1) is standard where x + n - 1 is even; aruco2 puts a
        # bottom tab under the others (x % 2 == n % 2).
        standard = [x for x in range(n) if (x + n - 1) % 2 == 0]
        column = min(standard, key=lambda x: abs(x - (n - 1) / 2))
        band = band_depth(self.square_size)
        x_centre = self.margin + (column + 0.5) * self.square_size
        y_centre = self.length - self.margin + band / 2
        return x_centre, y_centre, band * _LABEL_BAND_FRACTION

    def face_texture(self, face_index: int, draw_board_id: bool = True,
                     draw_edge_line: bool = True) -> np.ndarray:
        """
        Face ``face_index`` as a uint8 image of ``draw_res`` pixels.

        Rasterised from :meth:`face_rectangles`, with, optionally, the face's
        edge line -- the outline a printed net is cut along -- and its number.
        """
        texture = np.full(self.draw_res, 255, dtype=np.uint8)
        if draw_edge_line:
            # Ccube's make_blank_square, but never a line zero pixels wide:
            # there ``canvas[:, -0:] = 0`` blacks out the whole face, which
            # any draw_res below 1 / line_fraction would do.
            line = max(1, int(self.draw_res[0] * self.line_fraction))
            texture[:, :line] = 0
            texture[:line, :] = 0
            texture[:, -line:] = 0
            texture[-line:, :] = 0
        px_per_m = self.draw_res[0] / self.length
        rasterise_rectangles(self.face_rectangles(face_index), px_per_m, image=texture)
        if draw_board_id:
            x_centre, y_centre, height = self.face_label_anchor()
            text = f"{face_index}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            height_px = height * px_per_m
            (_, unit_h), _ = cv2.getTextSize(text, font, 1.0, 1)
            thickness = max(1, int(round(height_px / 10)))
            # Scaled so the digit, stroke included, is the height asked for.
            scale = max(height_px - thickness, 1) / unit_h
            (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
            origin = (int(round(x_centre * px_per_m - w / 2)),
                      int(round(y_centre * px_per_m + h / 2)))
            cv2.putText(texture, text, origin, font, scale, 0, thickness, cv2.LINE_AA)
        return texture

    # -- detection ------------------------------------------------------------

    def find_in_image(self, image, draw=False, camera: Camera | None = None,
                      wait_len=1) -> ImageDetection:
        """
        Detects the cube in an image, one face at a time.

        :param image: the image, uint8 or an integral 0..255 float.
        :param draw: whether to draw the detected corners.
        :param camera: unused -- aruco2's grid-board detector does not take
            a camera model.
        :param wait_len: the wait time for the visualisation. -1 waits for
            user keypress.
        :return ImageDetection: the corners found, keyed ``[face, gid]``.
        :raises ValueError: for an image that is not uint8 or an integral
            0..255 float -- from :func:`~pyCamSet.calibration_targets
            .markers.aruco2._as_uint8_image`, the same check
            :func:`~pyCamSet.calibration_targets.markers.aruco2_gridboard
            .detect_grid_board_corners` runs on every face lookup below, so
            it is not reimplemented here (round-3 review, P2: a local copy
            of the same convertibility check used to duplicate it, and could
            silently drift from it -- e.g. missing the
            ``np.ascontiguousarray`` the shared helper applies).
        """
        image = _as_uint8_image(image)

        if draw:
            im_idea = image.copy()
            target_size = [640, 480]
            d_f = max(int(min(np.array(im_idea.shape[:2])/target_size)), 1)
            im_idea = downsample_valid(im_idea, d_f).astype(np.uint8)
            if im_idea.ndim == 2:
                im_idea = np.tile(im_idea[..., None], (1, 1, 3))

        n_corners = (self.n_points + 1) ** 2
        seen_keys = []
        seen_data = []
        # aruco2 finds one grid board per call, and a face is only found
        # under its own ids, so each face is looked for on its own.
        for idb, ids in enumerate(self.face_ids):
            c_ids, c_pts = detect_grid_board_corners(
                image, self.grid_size, self._aruco_dict_int, self.square_size,
                ids=ids)
            if c_ids is None:
                continue
            valid = (c_ids >= 0) & (c_ids < n_corners)
            c_ids, c_pts = c_ids[valid], c_pts[valid]
            for cid, corner in zip(c_ids, c_pts):
                seen_keys.append([idb, int(cid)])
                seen_data.append(corner)
            if draw and len(c_ids):
                aruco.drawDetectedCornersCharuco(
                    im_idea,
                    np.asarray(c_pts, dtype=np.float32).reshape(-1, 1, 2) / d_f,
                    np.asarray(c_ids, dtype=np.int32).reshape(-1, 1),
                )

        if draw:
            cv2.imshow('detections', im_idea)
            cv2.waitKey(wait_len)

        return ImageDetection(keys=seen_keys, image_points=seen_data)

    # -- drawing --------------------------------------------------------------

    def plot(self, return_scene=False):
        """
        Draws a 3D model of the calibration target using pyVista

        """
        scene = self.faceData.draw_meshes(self.base_face, self.textures, return_scene=return_scene)
        if return_scene:
            return scene

    # Where a face goes in the net is Ccube's business, and the same here.
    net_affine_for_face = Ccube.net_affine_for_face
    face_extent = Ccube.face_extent
    apply_affine_xy = staticmethod(Ccube.apply_affine_xy)

    def save_to_pdf(
            self,
            f_out: Path | str,
            border_width: float = 10,
            individual_faces=False,
            data_format: str = "raster",
            draw_cut_outline: bool = True,
            draw_board_ids: bool = True,
    ) -> Path:
        """
        Write the cube's net, or its faces, as a PDF.

        :param f_out: where to write it, given a ``.pdf`` suffix; a default
            name when None.
        :param border_width: the white margin around the net or each face,
            in millimetres.
        :param individual_faces: write one face per page, each rasterised
            from its texture at the textures' own resolution, rather than
            one net. The pages are raster whatever ``data_format`` says, and
            asking for ``"vector"`` logs a warning.
        :param data_format: ``"vector"`` for cairo's rendering of
            :meth:`save_to_svg`, ``"raster"`` for the net of the face
            textures.
        :param draw_cut_outline: draw each face's outline to cut along.
        :param draw_board_ids: draw each face's number, in its band.
        :return: the file written.
        """
        f_out = export_path(f_out, ".pdf")
        blank_f = int(border_width * 0.0393701 * self.dpi)
        # The textures as built carry both the outline and the number; any
        # other choice is drawn afresh rather than silently ignored.
        if draw_cut_outline and draw_board_ids:
            textures = self.textures
        else:
            textures = [self.face_texture(k, draw_board_id=draw_board_ids,
                                          draw_edge_line=draw_cut_outline)
                        for k in range(6)]

        if individual_faces:
            if data_format == "vector":
                # Said, not silently swapped: a vector PDF was asked for.
                logger.warning(
                    "Ccube2 face-per-page PDFs are raster; %s is written from "
                    "the face textures, not as vector.", f_out)
            # Ccube writes each face to a PNG of its own; one page per face in
            # the one file asked for keeps save_printable's promise of a path.
            pages = []
            for face in textures:
                full_im = np.full(np.array(face.shape) + blank_f * 2, 255, dtype=np.uint8)
                full_im[blank_f:blank_f + face.shape[0], blank_f:blank_f + face.shape[1]] = face
                pages.append(Image.fromarray(full_im))
            pages[0].save(fp=f_out, resolution=self.dpi, save_all=True,
                          append_images=pages[1:])
            for page in pages:
                page.close()
            logger.info("Saved Ccube2 face-per-page PDF: %s", f_out)
            return f_out

        if data_format == "vector":
            from pyCamSet.utils.cairo_dll_helper import cairosvg_or_explain
            cairosvg = cairosvg_or_explain()
            svg_out = f_out.with_suffix(".svg")
            self.save_to_svg(
                f_out=svg_out,
                border_width=border_width,
                draw_cut_outline=draw_cut_outline,
                draw_board_ids=draw_board_ids,
                suppress_svg_log=True,
            )
            cairosvg.svg2pdf(url=str(svg_out), write_to=str(f_out))
            logger.info("Saved Ccube2 Vector PDF: %s", f_out)
            return f_out

        if data_format != "raster":
            raise ValueError("data_format must be one of: raster, vector")

        im_board = self.faceData.draw_net(textures, NET_FORMS)
        full_im = np.full(np.array(im_board.shape) + blank_f * 2, 255, dtype=np.uint8)
        full_im[blank_f:blank_f + im_board.shape[0],
                blank_f:blank_f + im_board.shape[1]] = im_board
        with Image.fromarray(full_im) as im:
            im.save(fp=f_out, resolution=self.dpi)
        logger.info("Saved Ccube2 Raster PDF: %s", f_out)
        return f_out

    def save_to_svg(
            self,
            f_out: Path | str,
            border_width: float = 10,
            draw_cut_outline: bool = True,
            draw_board_ids: bool = True,
            suppress_svg_log: bool = False,
    ) -> Path:
        """
        Write the cube's net as a true-vector SVG, at true mm scale.

        Each face is one ``<path>`` of rectangle sub-paths, placed in the net
        as Ccube places its faces, so abutting cells rasterise without seams.
        There is no embedded image.

        :param f_out: where to write it; a default name when None, and inside
            it when it is a directory.
        :param border_width: the white margin around the net, in millimetres.
        :param draw_cut_outline: draw each face's outline to cut along.
        :param draw_board_ids: draw each face's number, in its band.
        :param suppress_svg_log: skip the "Saved" log line.
        """
        f_out = export_path(f_out, ".svg")

        face_w, face_h = self.face_extent()
        local_outline = np.array(
            [[0.0, 0.0], [face_w, 0.0], [face_w, face_h], [0.0, face_h]],
            dtype=float,
        )
        affines = [self.net_affine_for_face(k) for k in range(6)]
        face_outlines_global = [self.apply_affine_xy(local_outline, A) for A in affines]

        # The page is the net's extent plus the border, in metres, as Ccube's
        # SVG is: the viewBox is in metres and width/height carry the mm.
        all_pts = np.vstack(face_outlines_global)
        min_xy = all_pts.min(axis=0) - border_width * 0.001
        max_xy = all_pts.max(axis=0) + border_width * 0.001
        canvas_w = round(float(max_xy[0] - min_xy[0]), 8)
        canvas_h = round(float(max_xy[1] - min_xy[1]), 8)
        offset = -min_xy

        dwg = svgwrite.Drawing(
            str(f_out),
            size=(f"{canvas_w * 1000.0:.6f}mm", f"{canvas_h * 1000.0:.6f}mm"),
            viewBox=f"0 0 {canvas_w:.8f} {canvas_h:.8f}",
        )
        dwg.add(dwg.rect(insert=(0, 0), size=(canvas_w, canvas_h), fill="white"))

        for face_idx, A in enumerate(affines):
            dwg.add(dwg.path(
                d=rectangles_svg_path(
                    self.face_rectangles(face_idx), affine=A, offset=offset),
                fill="black", stroke="none", fill_rule="nonzero",
            ))

        if draw_cut_outline:
            cut_line_w = float(self.length) * float(self.line_fraction)
            for poly in face_outlines_global:
                p = poly + offset
                dwg.add(
                    dwg.polygon(
                        points=[tuple(xy) for xy in p],
                        fill="none",
                        stroke="black",
                        stroke_width=cut_line_w,
                    )
                )

        if draw_board_ids:
            x_centre, y_centre, height = self.face_label_anchor()
            font_size = height / _ARIAL_DIGIT_HEIGHT
            # The baseline, so the digit's middle is on the band's middle.
            baseline = np.array([[x_centre, y_centre + height / 2]], dtype=float)
            for face_idx, A in enumerate(affines):
                p = self.apply_affine_xy(baseline, A)[0] + offset
                angle_deg = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
                t = dwg.text(
                    str(face_idx),
                    insert=(float(p[0]), float(p[1])),
                    fill="black",
                    font_size=f"{font_size:.8f}",
                    font_family="Arial",
                    font_weight="bold",
                    text_anchor="middle",
                )
                t.rotate(angle_deg, center=(float(p[0]), float(p[1])))
                dwg.add(t)

        svg_text = dwg.tostring()
        with open(f_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg_text)
            fh.flush()
            os.fsync(fh.fileno())

        if (not f_out.exists()) or f_out.stat().st_size == 0:
            raise IOError(f"SVG write failed: {f_out}")

        if not suppress_svg_log:
            logger.info("Saved Ccube2 SVG: %s", f_out)
        return f_out
