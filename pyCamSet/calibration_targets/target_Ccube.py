from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable
import numpy as np
from cv2 import aruco
import cv2
from PIL import Image
import svgwrite
from matplotlib import pyplot as plt
from tqdm import tqdm

from pyCamSet.calibration_targets import AbstractTarget, ImageDetection, FaceToShape
from pyCamSet.calibration_targets.charuco_detection import (
    build_charuco_detector_components,
    construct_charuco_detector,
)
from pyCamSet.cameras import Camera
from pyCamSet.utils.general_utils import split_aruco_dictionary, make_4x4h_tform, downsample_valid

# TFORMS = [
# 	([-1.209,-1.209, 1.209],[ 0.5,-0.5, 0.5]),
# 	([ 1.209,-1.209, 1.209],[ 0.5, 0.5,-0.5]),
# 	([0.   ,0.   ,1.571],[ 0.5,-0.5,-0.5]),
# 	([2.221,0.   ,2.221],[-0.5, 0.5,-0.5]),
# 	([3.142,0.   ,0.   ],[-0.5, 0.5, 0.5]),
# 	([-1.571, 0.   , 0.   ],[-0.5,-0.5, 0.5]),
# ]

# TFORMS Purpose - these are the transforms to get from the local coordinates of each face
# to the global coordinates of the cube. They are in the form of (rotation, translation),
# where rotation is in radians and translation is in the same units as the length of the
# cube (m). The order of the faces is: front, right, back, left, top, bottom. The
# rotations are around the x, y, and z axes respectively.
TFORMS = [
    (([2.22144147, 2.22144147, 0.        ]), ([-0.5, -0.5,  0.5])),
    (([-1.57079633,  0.        ,  0.        ]), ([-0.5, -0.5,  0.5])),
    (([-1.20919958, -1.20919958,  1.20919958]), ([ 0.5, -0.5,  0.5])),
    (([ 0.        ,  2.22144147, -2.22144147]), ([0.5, 0.5, 0.5])),
    (([0.        , 0.        , 1.57079633]), ([ 0.5, -0.5, -0.5])),
    (([1.20919958, 1.20919958, 1.20919958]), ([-0.5, -0.5, -0.5])),
]

# NET_FORMS Purpose - These are the transforms to get from the local coordinates of each
# face to the 3D coordinates of the net layout.
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

class Ccube(AbstractTarget):
    """
    This class defines a calibration target that consists of a Cube of ChArUco boards.
    """

    def __init__(self, length=20, n_points=5,
                 aruco_dict=aruco.DICT_4X4_1000,
                 draw_res=(1000, 1000),
                 border_fraction=0.1,
                 line_fraction=0.003,
                 legacy=False,
                 detection_options: dict | None = None,
                 ):
        super().__init__(inputs=locals())
        self.input_border_fraction = border_fraction
        self.actual_border_fraction = None
        self.line_fraction = line_fraction
        self.aruco_dict = aruco_dict
        self.length = length/1000
        self.square_size = self.length * (1 - border_fraction) / n_points
        if n_points % 2 == 0:
            split = int(n_points ** 2 / 2)
        else:
            split = int((n_points - 1) * (n_points + 1) / 2)
        self.markers_per_face = split
        self.a_dicts = split_aruco_dictionary(split, self.aruco_dict)
        if len(self.a_dicts) < 6:
            raise ValueError("Input dictionary of marker didn't contain enough "
                             "markers for this cube")

        self.boards = [
            aruco.CharucoBoard(
                (n_points, n_points), self.square_size,
                markerLength=0.75 * self.square_size,
                dictionary=a_dict,
                # legacy=True,
            )
            for a_dict in self.a_dicts][:6] #only need 6 of them!
        if legacy:
            [b.setLegacyPattern(True) for b in self.boards]
        self.detection_options = detection_options or {}  # Store the shared ChArUco detector overrides.
        self.detection_params, self.detector_params, self.refine_params = build_charuco_detector_components(
            self.detection_options
        )  # Build one shared parameter bundle for all six faces.

        self.n_points = n_points
        self.draw_res = draw_res
        self.dpi = self.draw_res[0] / self.length / 39.3701  # inch conversion
        blank_face, board_offset = make_blank_square(draw_res, line_fraction, border_fraction)
        sub_res = (draw_res[0] - 2 * board_offset, draw_res[1] - 2*board_offset)
        self.textures = [blank_face.copy() for _ in range(6)]
        # debug_t = []
        for idb, (t, board) in enumerate(zip(self.textures, self.boards)):
            t[board_offset:-board_offset, board_offset:-board_offset] = board.generateImage(sub_res)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = int(t.shape[0]/500)
            cv2.putText(t, f"{idb}", (t.shape[0]//100, t.shape[0]//100 * 99 ), font, font_scale, 0, thickness)

            # debug_t.append(board.draw(draw_res)) #DEBUG
        # self.textures = debug_t

        bd = np.array([board.getChessboardCorners() for board in self.boards])
        coord_bump = self.length*border_fraction/2
        board_coords = bd + np.array([coord_bump, coord_bump, 0])
        # board_coords = bd #DEBUG
        self.base_face = np.array([
                            [0, self.length,0],
                            [self.length, self.length,0],
                            [self.length, 0,0],
                            [0, 0,0],
                        ])

        # breakpoint()
        self.faceData = FaceToShape(
            face_local_coords=board_coords,
            face_transforms=[make_4x4h_tform(*t) for t in TFORMS],
            scale_factor= self.length,
        )
        self.point_data = self.faceData.point_data
        self._process_data()

        self.board_detectors = None
        self.given_legacy_warning = False

    def plot(self, return_scene = False):
        """
        Draws a 3D model of the calibration target using pyVista

        """
        scene = self.faceData.draw_meshes(self.base_face, self.textures, return_scene=return_scene)
        if return_scene:
            return scene

    def save_to_pdf(
            self,
            f_out: Path | None = None,
            border_width: float = 10,
            individual_faces=False,
            data_format: str = "raster",
    ):
        if individual_faces:
            for idf, face in enumerate(tqdm(self.textures)):
                blank_f = int(border_width * 0.0393701 * self.dpi)
                dims = np.array(face.shape) + blank_f * 2
                full_im = np.ones((dims)) * 255
                full_im[blank_f:-blank_f, blank_f:-blank_f] = face

                if f_out is None:
                    f_out = Path(f'Ccube_length_{self.length * 1000:.2f}mm' \
                                 f'_{self.n_points}_points_at' \
                                 f'_{self.square_size * 1000:.2f}mm_face_{idf}.png')
                full_im = full_im.astype(np.uint8)
                with Image.fromarray(full_im) as im:
                    im.save(fp=f_out, resolution=self.dpi)
                f_out = None
            return

        if data_format == "vector":
            try:
                import cairosvg
            except OSError as _cairo_err:
                raise OSError(
                    f"{_cairo_err}\n\n"
                    "pyCamSet's ChArUco/Ccube target code requires the native 'cairo' "
                    "library, which cairosvg needs but pip cannot install reliably on "
                    "Windows.\n"
                    "Fix: if using conda, run:\n"
                    "    conda install -c conda-forge cairo\n"
                    "Then try importing pyCamSet again."
                ) from _cairo_err
            if f_out is None:
                f_out = Path(
                    f'Ccube_length_{self.length * 1000:.2f}mm'
                    f'_{self.n_points}_points_at'
                    f'_{self.square_size * 1000:.2f}mm.pdf'
                )
            else:
                f_out = Path(f_out)
            svg_out = f_out.with_suffix(".svg")
            self.save_to_svg(
                f_out=svg_out,
                border_width=border_width,
                draw_cut_outline=True,
                draw_board_ids=True,
                suppress_svg_log=True,
            )
            cairosvg.svg2pdf(url=str(svg_out), write_to=str(f_out))
            logging.info("Saved Ccube Vector PDF: %s", f_out)
            return

        im_board = self.faceData.draw_net(self.textures, NET_FORMS)

        blank_f = int(border_width * 0.0393701 * self.dpi)
        dims = np.array(im_board.shape) + blank_f * 2
        full_im = np.ones((dims)) * 255
        full_im[blank_f:-blank_f, blank_f:-blank_f] = im_board

        if f_out is None:
            f_out = f'Ccube_length_{self.length * 1000:.2f}mm' \
                    f'_{self.n_points}_points_at' \
                    f'_{self.square_size * 1000:.2f}mm.pdf'
        full_im = full_im.astype(np.uint8)
        with Image.fromarray(full_im) as im:
            im.save(fp=f_out, resolution=self.dpi)

    # SVG export pipeline
    def charuco_layout_per_face(self) -> list[dict]:
        """
        Returns per-face board layout descriptors needed for vector reconstruction of each face.
        Each dict is expected to include: n_cols, n_rows, square, marker, dictionary.
        This method isolates project-specific board bookkeeping from generic SVG writing logic below.
        """
        # Each entry corresponds to one face and contains all parameters needed to reconstruct that face's geometry in vector format.
        layouts: list[dict] = []
        square_m = float(self.square_size)
        marker_m = float(self.square_size * 0.75)
        board_offset_m = float(self.length * self.input_border_fraction * 0.5)
        for board in self.boards:
            n_cols = int(board.getChessboardSize()[0]) # Number of chess squares in x direction
            n_rows = int(board.getChessboardSize()[1]) # Number of chess squares in y direction
            dct = board.getDictionary() # OpenCV ArUco dictionary object used by this face
            ids = np.asarray(board.getIds()).reshape(-1).astype(int) # List of marker ids on this face, as 1D array for easier indexing
            layouts.append({
                "n_cols": n_cols,
                "n_rows": n_rows,
                "square": square_m,
                "marker": marker_m,
                "board_offset": board_offset_m,
                "dictionary": dct,
                "board": board,
            })
        # Returns collected per-face descriptors for downstream vector export.
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
        # NET_FORMS is defined in row/col convention for draw_net; convert to x/y for polygon geometry.
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
        # Converts the stored face side length to float to avoid downstream dtype surprises in numpy/svg operations.
        side_m = float(self.length)  # Uses cube face side length in metres, consistent with class geometry units.
        return side_m, side_m  # Returns full face width and height in metres; Ccube faces are modelled as squares of side self.length.

    @staticmethod
    def aruco_bits_for_id(dictionary: cv2.aruco.Dictionary, marker_id: int) -> np.ndarray:
        # Reads marker bit resolution (e.g., 4 for DICT_4X4_* dictionaries) from dictionary metadata.
        marker_size = int(dictionary.markerSize)  # This determines the expected output bit-grid dimensions.

        # Retrieves raw bytes entry for one marker from dictionary storage.
        raw = np.asarray(dictionary.bytesList[marker_id])  # Python binding shape varies across OpenCV versions/builds.

        # Flattens raw bytes to one-dimensional array to remove binding-specific extra dimensions/channels.
        raw_flat = raw.reshape(-1)  # Produces a contiguous linear byte representation independent of original shape.

        # Computes expected packed byte count for one marker bit pattern (ceil(marker_size^2 / 8)).
        expected_n = int((marker_size * marker_size + 7) // 8)  # Matches C++ getBitsFromByteList byte-count rule.

        # Trims or pads byte vector to expected length to satisfy OpenCV assertion constraints.
        if raw_flat.size < expected_n:  # Handles rare cases where binding returns fewer bytes than expected.
            pad = np.zeros(expected_n - raw_flat.size, dtype=raw_flat.dtype)  # Creates zero padding bytes.
            raw_flat = np.concatenate([raw_flat, pad])  # Extends byte vector to required length.
        elif raw_flat.size > expected_n:  # Handles common cases where extra channel/copy bytes are present.
            raw_flat = raw_flat[:expected_n]  # Keeps only canonical packed marker payload.

        # Formats bytes as 1xN uint8 matrix as required by Dictionary_getBitsFromByteList.
        byte_list = raw_flat.astype(np.uint8).reshape(1, -1)  # Ensures dtype/shape align with OpenCV API expectations.

        # Tries direct packed-byte decoding first (fast path).
        try:
            bits = cv2.aruco.Dictionary_getBitsFromByteList(byte_list,
                                                            marker_size)  # Decodes packed bytes to marker bit grid.
            bits = np.asarray(bits, dtype=np.uint8)  # Normalises output type for consistent downstream processing.
            if bits.shape == (marker_size, marker_size):  # Validates expected bit-grid shape.
                return bits  # Returns successful direct decode.
        except cv2.error:
            pass  # Falls back to image-threshold decoding when packed-byte API path fails on this build.

        # Fallback path: render marker image and convert modules to bits by block averaging.
        side = marker_size * 20  # Chooses render size as integer multiple of marker_size for clean module partitioning.
        marker_img = np.zeros((side, side), dtype=np.uint8)  # Allocates image buffer for generated marker.
        cv2.aruco.generateImageMarker(dictionary, int(marker_id), side, marker_img,
                                      1)  # Renders marker with one-bit border.

        # Removes one-cell border introduced by generator to isolate the marker payload region.
        cell = side // (marker_size + 2)  # Computes nominal module pixel width including border.
        inner = marker_img[cell:-cell, cell:-cell]  # Crops border, leaving marker_size x marker_size payload grid area.

        # Re-estimates payload cell size after crop to avoid accumulation of integer rounding error.
        inner_cell = inner.shape[0] // marker_size  # Computes pixel width per payload module.

        # Allocates output bit matrix.
        out = np.zeros((marker_size, marker_size), dtype=np.uint8)  # Stores decoded binary modules.

        # Decodes each payload module by average intensity threshold.
        for r in range(marker_size):  # Iterates payload rows.
            for c in range(marker_size):  # Iterates payload columns.
                block = inner[
                    r * inner_cell:(r + 1) * inner_cell, c * inner_cell:(c + 1) * inner_cell]  # Extracts module block.
                out[r, c] = 1 if float(block.mean()) < 127.5 else 0  # Black module -> 1, white module -> 0.

        # Returns fallback-decoded bit matrix.
        return out

    @staticmethod
    def aruco_marker_grid_for_id(dictionary: cv2.aruco.Dictionary, marker_id: int) -> np.ndarray:
        """Returns full marker grid (payload + one-cell border), with 1=black and 0=white."""
        marker_size = int(dictionary.markerSize)
        n_cells = marker_size + 2
        cell_px = 24
        side = n_cells * cell_px

        marker_img = np.zeros((side, side), dtype=np.uint8)
        cv2.aruco.generateImageMarker(dictionary, int(marker_id), side, marker_img, 1)

        grid = np.zeros((n_cells, n_cells), dtype=np.uint8)
        for r in range(n_cells):
            for c in range(n_cells):
                block = marker_img[r * cell_px:(r + 1) * cell_px, c * cell_px:(c + 1) * cell_px]
                grid[r, c] = 1 if float(block.mean()) < 127.5 else 0
        return grid

    @staticmethod
    def apply_affine_xy(pts_xy: np.ndarray, A: np.ndarray) -> np.ndarray:
        # Appends homogeneous coordinate 1 to each point so affine transform can be applied by matrix multiply.
        pts_h = np.c_[pts_xy, np.ones((pts_xy.shape[0], 1), dtype=float)]  # Shape: (N,3) for homogeneous coordinates.
        # Applies affine matrix and drops homogeneous dimension, yielding transformed x,y points.
        return (A @ pts_h.T).T[:, :2]  # Shape: (N,2) in target coordinate system.

    def save_to_svg(
            self,
            f_out: Path | str | None = None,
            border_width: float = 10,
            draw_cut_outline: bool = True,
            draw_board_ids: bool = True,
            suppress_svg_log: bool = False,
    ) -> Path:
        default_name = (
            f"Ccube_length_{self.length * 1000:.2f}mm_"
            f"{self.n_points}_points_at_{self.square_size * 1000:.2f}mm_true_vector.svg"
        )

        raw_f_out = f_out
        if isinstance(f_out, str):
            f_out = Path(f_out)

        if f_out is None:
            f_out = Path.cwd() / default_name
        else:
            # If user passed a directory, write default file inside it.
            raw_s = str(raw_f_out) if raw_f_out is not None else ""
            looks_like_dir = raw_s.endswith(("/", "\\"))
            if (f_out.exists() and f_out.is_dir()) or looks_like_dir:
                f_out = f_out / default_name

        f_out = f_out.expanduser().with_suffix(".svg").resolve()
        f_out.parent.mkdir(parents=True, exist_ok=True)

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
                grid = self.aruco_marker_grid_for_id(dct, marker_id)  # 1=black, 0=white
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
            logging.info("Saved Ccube SVG: %s", f_out)
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

        if self.board_detectors is None:
            self.board_detectors = [
                construct_charuco_detector(
                    board,
                    self.detection_params,
                    self.detector_params,
                    self.refine_params,
                )  # Reuse the shared constructor/fallback path for each face detector.
                for board in self.boards
            ]


        if draw:
            im_idea = image.copy()
            target_size = [640, 480]
            d_f = max(int(min(np.array(im_idea.shape[:2])/target_size)), 1)
            im_idea = downsample_valid(im_idea, d_f).astype(np.uint8)
            if im_idea.ndim == 2:
                im_idea = np.tile(im_idea[..., None], (1, 1, 3))


        seen_keys = []
        seen_data = []
        for idb, bd in enumerate(self.board_detectors):
            c_corners, c_ids, mloc, mid =  bd.detectBoard(image)
            if c_corners is None and mloc is not None:
                if not self.given_legacy_warning:
                    logging.warning("Found markers, but no corners, trying using alternative board detection")
                    self.given_legacy_warning = True
                am_legacy = self.boards[idb].getLegacyPattern()
                self.boards[idb].setLegacyPattern(not am_legacy)
                c_corners, c_ids, mloc, mid = bd.detectBoard(image, markerCorners=mloc, markerIds=mid)

            if c_ids is not None:
                for cid, corner in zip(c_ids[:, 0], c_corners[:, 0, :]):
                    seen_keys.append([idb, cid])
                    seen_data.append(corner)
                if draw:
                    aruco.drawDetectedCornersCharuco(im_idea, c_corners/d_f, c_ids)

        if draw:
            cv2.imshow('detections', im_idea)
            cv2.waitKey(wait_len)

        return ImageDetection(keys=seen_keys, image_points=seen_data)



if __name__ == '__main__':
    test = Ccube(n_points=7, length=4)
    test.plot()
    # test.get_printable_texture()
