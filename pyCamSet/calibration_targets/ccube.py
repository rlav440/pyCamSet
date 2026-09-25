from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Iterable
import numpy as np
from cv2 import aruco
import cv2
from PIL import Image
import svgwrite
from matplotlib import pyplot as plt
from tqdm import tqdm

from pyCamSet.calibration_targets.core import (
    AbstractTarget, ImageDetection, FaceToShape, exclude_by_prefix,
)
from pyCamSet.calibration_targets.markers.aruco_opencv import marker_bit_grid
from pyCamSet.calibration_targets.markers.backend_registry import (
    ARUCO1_BACKEND,
    dict_names_for_backend,
    dictionary_id,
)
from pyCamSet.calibration_targets.core.abstract_target import (
    EXPORT_SUFFIXES, export_path)
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
from pyCamSet.calibration_targets.markers.aruco_opencv import ARUCO_OPENCV_DETECTOR
from pyCamSet.calibration_targets.markers.legacy_probe import should_warn_legacy_mismatch
from pyCamSet.cameras import Camera
from pyCamSet.utils.general_utils import split_aruco_dictionary, make_4x4h_tform, downsample_valid

#: The smallest face a cube can have.  Two is where OpenCV stops building a
#: board at all -- below it OpenCV does not raise a catchable error, it
#: corrupts its own module state -- and three is where a face has more than
#: the single chessboard corner that squeezes down to a bare point.
_MIN_POINTS = 3

#: The dictionary a cube is printed with unless another is asked for.  It is
#: split six ways, so it needs markers to spare.
_DEFAULT_DICT_NAME = "DICT_4X4_1000"


# Local face coordinates to cube coordinates, as (rotation in radians,
# translation in cube lengths), for front, right, back, left, top, bottom.
TFORMS = [
    (([2.22144147, 2.22144147, 0.        ]), ([-0.5, -0.5,  0.5])),
    (([-1.57079633,  0.        ,  0.        ]), ([-0.5, -0.5,  0.5])),
    (([-1.20919958, -1.20919958,  1.20919958]), ([ 0.5, -0.5,  0.5])),
    (([ 0.        ,  2.22144147, -2.22144147]), ([0.5, 0.5, 0.5])),
    (([0.        , 0.        , 1.57079633]), ([ 0.5, -0.5, -0.5])),
    (([1.20919958, 1.20919958, 1.20919958]), ([-0.5, -0.5, -0.5])),
]

# Local face coordinates to the printable net's coordinates.
NET_FORMS=[
	[[1.0,0.0,0.0], [0.0,1.0,0.0],[0.0,0.0,1.0]],
	[[0.0,1.0,0.0], [-1.0,0.0,0.0],[0.0,0.0,1.0]],
	[[1.0,0.0,1.0], [0.0,1.0,0.0],[0.0,0.0,1.0]],
	[[0.0,-1.0,1.0], [1.0,0.0,1.0],[0.0,0.0,1.0]],
	[[1.0,0.0,2.0], [0.0,1.0,0.0],[0.0,0.0,1.0]],
	[[1.0,0.0,-1.0], [0.0,1.0,0.0],[0.0,0.0,1.0]],
]

def make_blank_square(draw_res, line_fraction, border_fraction):
    """
    This function makes a blank face of a square, with surrounding edge lines set to black.
    For convenience, it also returns the array offset used for the border fraction.
    
    :param draw_res: the drawing resolution of the cube face
    :param line_fraction: the thickness of the line as a fraction of the face width
    :param border_fraction: the size of the fraction to use as the border of the cube.
    """
    canvas = np.ones(draw_res)*255
    int_line = int(draw_res[0] * line_fraction)
    canvas[:, :int_line] = 0
    canvas[:int_line, :] = 0
    canvas[:, -int_line:] = 0
    canvas[-int_line:, :] = 0
    return canvas, int(border_fraction * draw_res[0]/2)

#: aruco1 and aruco2 only disagree on these four dictionaries, so neither is
#: offered as a construction choice -- a saved run naming one still loads,
#: since build_target() reads a spec's raw dict rather than going through
#: choices.  aruco2's two ALVAR dictionaries are not offered either: the
#: choices are aruco1's, whichever detector later reads the target.
_UNOFFERED_DICT_PREFIXES = ("DICT_APRILTAG_",)

class Ccube(AbstractTarget):
    """
    This class defines a calibration target that consists of a Cube of ChArUco boards.
    """

    DETECTOR_BACKENDS = {
        "aruco1": ARUCO_OPENCV_DETECTOR,
        "aruco2": ARUCO2_DETECTOR,
    }

    @classmethod
    def construction_parameters(cls, backend: str | None = None) -> Parameterisation:
        """
        What decides where a cube's corners are, and how it prints.

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
        """How a Ccube net is drawn, which is not what it is."""
        return DocumentedParameters(
            cls.save_printable, "border_width", "draw_cut_outline",
            "draw_board_ids", "individual_faces")

    @classmethod
    def printable_name(cls, values: dict, kind: str = "svg") -> str:
        return (f"ccube_{int(values['n_points'])}points_"
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
            a page. Suggested: off.
        :raises ValueError: for a format a cube cannot be written as
        """
        if kind == "svg":
            return self.save_to_svg(
                path, border_width=border_width,
                draw_cut_outline=draw_cut_outline, draw_board_ids=draw_board_ids)
        if kind in ("pdf_vector", "pdf_raster"):
            return self.save_to_pdf(
                path, border_width=border_width,
                individual_faces=individual_faces,
                data_format="vector" if kind == "pdf_vector" else "raster")
        raise ValueError(f"A Ccube cannot be written as {kind!r}.")

    def __init__(self, length: float = 20.0, n_points: int = 5,
                 aruco_dict=_DEFAULT_DICT_NAME,
                 draw_res=(1000, 1000),
                 border_fraction: float = 0.1,
                 line_fraction: float = 0.003,
                 legacy: bool = False,
                 marker_backend: str = "aruco1",
                 detection_options: dict | None = None,
                 ):
        """
        Initialises a cube whose six faces are each a ChArUco board.

        :param length: Cube edge (mm) -- the printed edge length of the
            cube, in millimetres, border included. Suggested: 20-200.
        :param n_points: Squares per face -- chessboard squares along one
            edge of one of the cube's six faces. Suggested: 4-8.
        :param aruco_dict: ArUco dictionary -- the marker alphabet, split
            six ways so that each face carries markers of its own.
        :param draw_res: the resolution each face texture is drawn at.
        :param border_fraction: Border fraction -- how much of each face is
            blank margin rather than board. Detection: too little and
            markers near an edge are cut by the fold. Suggested: 0.05-0.15.
        :param line_fraction: the thickness of a face's edge line, as a
            fraction of the face width.
        :param legacy: Legacy pattern -- which of OpenCV's two marker
            layouts the faces were printed to; must match how the physical
            cube was printed, since detection never switches patterns
            automatically. Detection: the wrong one finds every marker and
            no corners (and, for a face with an even number of rows, logs a
            warning naming the likely correct setting and the face).
            Suggested: off (OpenCV's current pattern, and the default),
            unless the cube predates OpenCV 4.6.
        :param marker_backend: the detector the cube is read with,
            "aruco1" (OpenCV) or "aruco2" (the aruco2 package). Chosen in the
            detection phase rather than when the cube is made: both print
            the offered dictionaries identically, so it says how the cube
            is read, not what it is. Defaults to "aruco1".
        :param detection_options: what the detector is told, by the keys
            :meth:`detector_parameterisation` describes.
        """
        super().__init__(inputs=locals(), backend=marker_backend)

        # A face is a ChArUco board, and OpenCV does not refuse one that
        # cannot exist -- it raises with an exception still set, aborting
        # the next call to build one.  How large the cube is is nobody's
        # business here; that it is a cube at all is.
        if n_points < _MIN_POINTS:
            raise ValueError(
                f"A Ccube face must be at least {_MIN_POINTS}x{_MIN_POINTS} "
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

        self.input_border_fraction = border_fraction
        self.actual_border_fraction = None
        self.line_fraction = line_fraction
        self.marker_backend = marker_backend
        # A cv2.aruco.Dictionary is also accepted, and is passed through.
        if isinstance(aruco_dict, (int, str)):
            self._aruco_dict_int = dictionary_id(aruco_dict, marker_backend)
            aruco_dict = self._aruco_dict_int
        else:
            self._aruco_dict_int = None
        self.aruco_dict = aruco_dict
        self.length = length/1000
        self.square_size = self.length * (1 - border_fraction) / n_points
        if n_points % 2 == 0:
            split = int(n_points ** 2 / 2)
        else:
            split = int((n_points - 1) * (n_points + 1) / 2)
        self.markers_per_face = split
        resolved_dict = resolve_dictionary(aruco_dict, marker_backend)
        if 6 * split > (held := int(resolved_dict.bytesList.shape[0])):
            # The only ceiling a cube has: six faces are cut from one marker
            # alphabet, and it ends.  Splitting past the end is a numpy error
            # about inhomogeneous shapes, several frames from here.
            raise ValueError(
                f"A {n_points}x{n_points} cube needs {6 * split} markers, six "
                f"faces of {split}, and this dictionary holds {held}.")
        self.a_dicts = split_aruco_dictionary(split, resolved_dict)

        self.boards = [
            aruco.CharucoBoard(
                (n_points, n_points), self.square_size,
                markerLength=0.75 * self.square_size,
                dictionary=a_dict,
            )
            for a_dict in self.a_dicts][:6]
        if legacy:
            [b.setLegacyPattern(True) for b in self.boards]
        self.n_points = n_points
        self.draw_res = draw_res
        self.dpi = self.draw_res[0] / self.length / 39.3701  # inch conversion
        blank_face, board_offset = make_blank_square(draw_res, line_fraction, border_fraction)
        sub_res = (draw_res[0] - 2 * board_offset, draw_res[1] - 2*board_offset)
        self.textures = [blank_face.copy() for _ in range(6)]
        for idb, (t, board) in enumerate(zip(self.textures, self.boards)):
            t[board_offset:-board_offset, board_offset:-board_offset] = board.generateImage(sub_res)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = int(t.shape[0]/500)
            # OpenCV 5's putText asserts an 8-bit image, and these textures are
            # float; label a uint8 copy and write it back to keep the dtype.
            labelled = t.astype(np.uint8)
            cv2.putText(labelled, f"{idb}", (t.shape[0]//100, t.shape[0]//100 * 99 ), font, font_scale, 0, thickness)
            t[...] = labelled

        bd = np.array([board.getChessboardCorners() for board in self.boards])
        coord_bump = self.length*border_fraction/2
        board_coords = bd + np.array([coord_bump, coord_bump, 0])
        self.base_face = np.array([
                            [0, self.length,0],
                            [self.length, self.length,0],
                            [self.length, 0,0],
                            [0, 0,0],
                        ])

        self.faceData = FaceToShape(
            face_local_coords=board_coords,
            face_transforms=[make_4x4h_tform(*t) for t in TFORMS],
            scale_factor= self.length,
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

    def _warn_legacy_once(self, idb: int, board) -> None:
        """Warn once per target when a frame on one face gives strong
        evidence that face's physical board uses the OTHER legacy pattern
        than the one this target is configured with. Shared wording with
        ChArUco's own ``_warn_legacy_once`` (charuco/target.py), plus the
        face index: detection never switches patterns itself, so this is a
        warning only -- corners are never taken from the other pattern (see
        markers.legacy_probe.should_warn_legacy_mismatch).

        Also stashes the message on ``self.legacy_warning_message`` -- see
        that attribute's docstring above.

        The ``given_legacy_warning`` check-then-set is done under
        ``self._legacy_warning_lock`` so concurrent callers on the same
        instance cannot both pass the check before either sets the flag
        (round-3 review, P2): only the first caller to acquire the lock
        actually builds and logs a message.

        :param idb: the face index (0-5) that triggered the warning, so the
            message names WHICH face looks mismatched.
        :param board: that face's own board (``self.boards[idb]``), read
            dynamically rather than hardcoded, but never mutated here.
        """
        with self._legacy_warning_lock:
            if self.given_legacy_warning:
                return
            configured = board.getLegacyPattern()
            likely = not configured
            msg = (
                f"Ccube: face {idb} images look like a legacy={likely} "
                f"board but this target is legacy={configured}; check the "
                f"target's legacy setting."
            )
            logger.warning(msg)
            self.given_legacy_warning = True
            self.legacy_warning_message = msg

    def plot(self, return_scene = False):
        """
        Draws a 3D model of the calibration target using pyVista

        """
        scene = self.faceData.draw_meshes(self.base_face, self.textures, return_scene=return_scene)
        if return_scene:
            return scene

    def save_to_pdf(
            self,
            f_out: Path | str,
            border_width: float = 10,
            individual_faces=False,
            data_format: str = "raster",
    ):
        if individual_faces:
            # One file per face, named from the path asked for: face 0 used
            # to take that path and the other five landed in the working
            # directory under a default name.
            base = export_path(f_out, ".png")
            for idf, face in enumerate(tqdm(self.textures)):
                blank_f = int(border_width * 0.0393701 * self.dpi)
                dims = np.array(face.shape) + blank_f * 2
                full_im = np.ones((dims)) * 255
                full_im[blank_f:-blank_f, blank_f:-blank_f] = face
                full_im = full_im.astype(np.uint8)
                with Image.fromarray(full_im) as im:
                    im.save(fp=base.with_name(f"{base.stem}_face_{idf}.png"),
                            resolution=self.dpi)
            return

        if data_format == "vector":
            from pyCamSet.utils.cairo_dll_helper import cairosvg_or_explain
            cairosvg = cairosvg_or_explain()
            f_out = export_path(f_out, ".pdf")
            svg_out = f_out.with_suffix(".svg")
            self.save_to_svg(
                f_out=svg_out,
                border_width=border_width,
                draw_cut_outline=True,
                draw_board_ids=True,
                suppress_svg_log=True,
            )
            cairosvg.svg2pdf(url=str(svg_out), write_to=str(f_out))
            logger.info("Saved Ccube Vector PDF: %s", f_out)
            return

        im_board = self.faceData.draw_net(self.textures, NET_FORMS)

        blank_f = int(border_width * 0.0393701 * self.dpi)
        dims = np.array(im_board.shape) + blank_f * 2
        full_im = np.ones((dims)) * 255
        full_im[blank_f:-blank_f, blank_f:-blank_f] = im_board

        f_out = export_path(f_out, ".pdf")
        full_im = full_im.astype(np.uint8)
        with Image.fromarray(full_im) as im:
            im.save(fp=f_out, resolution=self.dpi)

    def charuco_layout_per_face(self) -> list[dict]:
        """What each face is, for the vector export to redraw it from."""
        layouts: list[dict] = []
        square_m = float(self.square_size)
        marker_m = float(self.square_size * 0.75)
        board_offset_m = float(self.length * self.input_border_fraction * 0.5)
        for board in self.boards:
            n_cols = int(board.getChessboardSize()[0])
            n_rows = int(board.getChessboardSize()[1])
            dct = board.getDictionary()
            ids = np.asarray(board.getIds()).reshape(-1).astype(int)
            layouts.append({
                "n_cols": n_cols,
                "n_rows": n_rows,
                "square": square_m,
                "marker": marker_m,
                "board_offset": board_offset_m,
                "dictionary": dct,
                "board": board,
            })
        return layouts

    def iter_marker_slots_for_face(self, layout: dict) -> Iterable[dict]:
        board = layout["board"]
        ids = np.asarray(board.getIds()).reshape(-1).astype(int)
        obj_points = board.getObjPoints()
        board_offset = float(layout.get("board_offset", 0.0))

        if len(ids) != len(obj_points):
            raise ValueError(
                f"Mismatch between ids ({len(ids)}) and obj_points ({len(obj_points)}) in Charuco board."
            )

        for marker_id, corners in zip(ids, obj_points):
            c = np.asarray(corners, dtype=float)
            if c.ndim == 3:
                c = c.reshape(-1, c.shape[-1])

            if c.shape[0] != 4 or c.shape[1] < 2:
                raise ValueError(f"Unexpected marker corner shape: {c.shape}")

            xy = c[:, :2] + board_offset

            yield {
                "id": int(marker_id),
                "quad_xy": xy,
            }

    @staticmethod
    def _bilinear_quad(q: np.ndarray, u: float, v: float) -> np.ndarray:
        p00 = q[0]
        p10 = q[1]
        p11 = q[2]
        p01 = q[3]
        return (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + u * v * p11 + (1 - u) * v * p01

    def net_affine_for_face(self, face_index: int) -> np.ndarray:
        # NET_FORMS is in draw_net's row/col convention; polygons want x/y.
        f_rc = np.asarray(NET_FORMS[face_index], dtype=float)
        if f_rc.shape != (3, 3):
            raise ValueError(f"NET_FORMS[{face_index}] must be 3x3, got {f_rc.shape}.")

        permute = np.array(
            [[0.0, 1.0, 0.0],
             [1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )
        f_xy_units = permute @ f_rc @ permute

        A = f_xy_units.copy()
        A[:2, 2] *= float(self.length)
        return A

    def face_extent(self) -> tuple[float, float]:
        side_m = float(self.length)
        return side_m, side_m

    @staticmethod
    def apply_affine_xy(pts_xy: np.ndarray, A: np.ndarray) -> np.ndarray:
        """(N, 2) points through a 3x3 affine."""
        pts_h = np.c_[pts_xy, np.ones((pts_xy.shape[0], 1), dtype=float)]
        return (A @ pts_h.T).T[:, :2]

    def save_to_svg(
            self,
            f_out: Path | str,
            border_width: float = 10,
            draw_cut_outline: bool = True,
            draw_board_ids: bool = True,
            suppress_svg_log: bool = False,
    ) -> Path:
        f_out = export_path(f_out, ".svg")

        face_w, face_h = self.face_extent()
        black_polys_global: list[np.ndarray] = []
        face_outlines_global: list[np.ndarray] = []
        face_labels_global: list[tuple[str, np.ndarray, float]] = []

        layouts = self.charuco_layout_per_face()
        for face_idx, layout in enumerate(layouts):
            A = self.net_affine_for_face(face_idx)

            local_outline = np.array(
                [[0.0, 0.0], [face_w, 0.0], [face_w, face_h], [0.0, face_h]],
                dtype=float,
            )
            face_outlines_global.append(self.apply_affine_xy(local_outline, A))

            if draw_board_ids:
                # Match PDF intent: near bottom-left of each face.
                label_local = np.array([[0.01 * face_w, 0.99 * face_h]], dtype=float)
                label_xy = self.apply_affine_xy(label_local, A)[0]
                label_angle_deg = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
                face_labels_global.append((str(face_idx), label_xy, label_angle_deg))

            # Draw Charuco chess squares (OpenCV starts with black at top-left).
            n_cols = int(layout["n_cols"])
            n_rows = int(layout["n_rows"])
            sq = float(layout["square"])
            board_offset = float(layout["board_offset"])
            for r in range(n_rows):
                for c in range(n_cols):
                    if (r + c) % 2 != 0:
                        continue
                    x0 = board_offset + c * sq
                    y0 = board_offset + r * sq
                    chess_poly_local = np.array(
                        [[x0, y0], [x0 + sq, y0], [x0 + sq, y0 + sq], [x0, y0 + sq]],
                        dtype=float,
                    )
                    black_polys_global.append(self.apply_affine_xy(chess_poly_local, A))

            dct = layout["dictionary"]
            for slot in self.iter_marker_slots_for_face(layout):
                marker_id = int(slot["id"])
                q = np.asarray(slot["quad_xy"], dtype=float)  # (4,2): tl,tr,br,bl
                grid = marker_bit_grid(dct, marker_id)  # 1=black, 0=white
                n_rows, n_cols = grid.shape

                for r in range(n_rows):
                    v0 = r / n_rows
                    v1 = (r + 1) / n_rows
                    for c in range(n_cols):
                        if grid[r, c] != 1:
                            continue
                        u0 = c / n_cols
                        u1 = (c + 1) / n_cols

                        poly_local = np.array(
                            [
                                self._bilinear_quad(q, u0, v0),
                                self._bilinear_quad(q, u1, v0),
                                self._bilinear_quad(q, u1, v1),
                                self._bilinear_quad(q, u0, v1),
                            ],
                            dtype=float,
                        )
                        black_polys_global.append(self.apply_affine_xy(poly_local, A))

        if not black_polys_global:
            raise RuntimeError("No vector marker polygons were generated for SVG export.")

        all_pts = np.vstack(black_polys_global + face_outlines_global)
        min_xy = all_pts.min(axis=0)
        max_xy = all_pts.max(axis=0)

        min_xy -= border_width * 0.001
        max_xy += border_width * 0.001
        canvas_w = float(max_xy[0] - min_xy[0])
        canvas_h = float(max_xy[1] - min_xy[1])

        dwg = svgwrite.Drawing(
            str(f_out),
            size=(f"{canvas_w * 1000.0:.6f}mm", f"{canvas_h * 1000.0:.6f}mm"),
            viewBox=f"0 0 {canvas_w:.6f} {canvas_h:.6f}",
        )
        dwg.add(dwg.rect(insert=(0, 0), size=(canvas_w, canvas_h), fill="white"))

        offset = -min_xy
        for poly in black_polys_global:
            p = poly + offset
            dwg.add(dwg.polygon(points=[tuple(xy) for xy in p], fill="black", stroke="none"))

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
            label_size = float(face_h) * 0.045
            for label, xy, angle_deg in face_labels_global:
                p = xy + offset
                t = dwg.text(
                    label,
                    insert=(float(p[0]), float(p[1])),
                    fill="black",
                    font_size=f"{label_size:.6f}",
                    font_family="Arial",
                    font_weight="bold",
                )
                t.rotate(angle_deg, center=(float(p[0]), float(p[1])))
                dwg.add(t)

        svg_text = dwg.tostring()
        with open(f_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg_text)
            fh.flush()
            import os
            os.fsync(fh.fileno())

        if (not f_out.exists()) or f_out.stat().st_size == 0:
            raise IOError(f"SVG write failed: {f_out}")

        if suppress_svg_log==False:
            logger.info("Saved Ccube SVG: %s", f_out)
        return f_out

    def find_in_image(self, image, draw=False, camera: Camera|None = None, wait_len=1) -> ImageDetection:
        """
        An implementation of the find in image function for

        :param image: the image
        :param draw: whether to draw the visualisation of the cube,
        :param camera: an optional camera model, for more accurate detections.
        :param wait_len: the wait time for the visualisation. -1 waits for user keypress.

        Returns:

        """

        image = np.asarray(image)
        if image.dtype != np.uint8:
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
                    "Ccube detection requires a uint8 image or an integral floating-point image "
                    f"in the range 0..255; got dtype {image.dtype}."
                )
            image = image.astype(np.uint8)

        if self.board_detectors is None:
            # Built here rather than in __init__: six detectors are expensive
            # and a cube is often made only to be printed.
            self.board_detectors = [
                ARUCO_OPENCV_DETECTOR.build_detector(board, self.detection_options)
                for board in self.boards
            ]


        if draw:
            im_idea = image.copy()
            target_size = [640, 480]
            d_f = max(int(min(np.array(im_idea.shape[:2])/target_size)), 1)
            im_idea = downsample_valid(im_idea, d_f).astype(np.uint8)
            if im_idea.ndim == 2:
                im_idea = np.tile(im_idea[..., None], (1, 1, 3))


        if self.marker_backend == "aruco2":
            # Detect once with the parent dictionary, then interpolate per
            # face with that face's own board.
            markers = detect_markers(image, self._aruco_dict_int)
            seen_keys = []
            seen_data = []
            for idb, board in enumerate(self.boards):
                # global id -> (face, local id): face = gid // markers_per_face;
                # markers from other faces (or stray ids) are skipped.
                face_markers = [
                    (gid - idb * self.markers_per_face, corners)
                    for gid, corners in markers
                    if gid // self.markers_per_face == idb
                ]
                if not face_markers:
                    continue
                c_ids, c_pts = interpolate_board_corners(
                    image,
                    board,
                    face_markers,
                    # Bound through default args, so the warning names the
                    # face that failed rather than wherever the loop has
                    # reached. None once warned: the callee only pays for
                    # the opposite-pattern probe when it is given one.
                    warn_legacy=(
                        None if self.given_legacy_warning else
                        (lambda _i=idb, _b=board: self._warn_legacy_once(_i, _b))
                    ),
                )
                if c_ids is None:
                    continue
                for cid, corner in zip(c_ids, c_pts):
                    seen_keys.append([idb, int(cid)])
                    seen_data.append(corner)
                if draw:
                    aruco.drawDetectedCornersCharuco(
                        im_idea,
                        np.asarray(c_pts, dtype=np.float32).reshape(-1, 1, 2) / d_f,
                        np.asarray(c_ids, dtype=np.int32).reshape(-1, 1),
                    )
            if draw:
                cv2.imshow('detections', im_idea)
                cv2.waitKey(wait_len)
            return ImageDetection(keys=seen_keys, image_points=seen_data)


        seen_keys = []
        seen_data = []
        for idb, bd in enumerate(self.board_detectors):
            c_corners, c_ids, mloc, mid =  bd.detectBoard(image)
            if c_corners is None and mloc is not None:
                # A mismatch warns but never retries under the other
                # pattern, and never takes corners from it. The probe costs
                # a throwaway board, detector and detectBoard call, so it is
                # skipped once this target has warned.
                if (not self.given_legacy_warning) and should_warn_legacy_mismatch(
                    self.boards[idb], self.detection_options, image,
                    len(mloc), mloc, mid,
                ):
                    self._warn_legacy_once(idb, self.boards[idb])

            if c_ids is not None:
                # OpenCV 5 squeezes detectBoard's singleton axis; normalise.
                ids_flat = np.asarray(c_ids).reshape(-1)
                corners_flat = np.asarray(c_corners).reshape(-1, 2)
                for cid, corner in zip(ids_flat, corners_flat):
                    seen_keys.append([idb, cid])
                    seen_data.append(corner)
                if draw:
                    aruco.drawDetectedCornersCharuco(
                        im_idea,
                        np.asarray(c_corners).reshape(-1, 1, 2) / d_f,
                        np.asarray(c_ids).reshape(-1, 1),
                    )

        if draw:
            cv2.imshow('detections', im_idea)
            cv2.waitKey(wait_len)

        return ImageDetection(keys=seen_keys, image_points=seen_data)


