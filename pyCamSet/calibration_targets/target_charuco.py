from __future__ import annotations

import logging
from pathlib import Path

import cairosvg
import cv2
import numpy as np
import svgwrite
from PIL import Image
from cv2 import aruco
from matplotlib import pyplot as plt

from pyCamSet.calibration_targets.abstract_target import AbstractTarget
from pyCamSet.calibration_targets.target_detections import ImageDetection
from pyCamSet.cameras import Camera
from pyCamSet.utils.general_utils import downsample_valid


class ChArUco(AbstractTarget):
    def __init__(
        self,
        num_squares_x,
        num_squares_y,
        square_size,
        marker_fraction=0.8,
        a_dict=cv2.aruco.DICT_4X4_1000,
        legacy=False,
        detection_options: dict | None = None,
    ):
        """
        Initialises a ChArUco board in mm.

        :param num_squares_x: number of squares in the x direction
        :param num_squares_y: number of squares in the y direction
        :param square_size: the size of a square in mm! mm!
        :param marker_fraction: the percentage of a chessboard square occupied by a marker
        :param a_dict: the aruco dictionary to use.
        """
        super().__init__(inputs=locals())

        # define checker and marker size

        self.square_size = square_size / 1000
        marker_size = marker_fraction * self.square_size  # 80% of the square size
        # convert to meters

        # Create the dictionary for the Charuco board
        self.a_dict = cv2.aruco.getPredefinedDictionary(a_dict)
        # Create the Charuco board
        self.board = cv2.aruco.CharucoBoard((num_squares_x, num_squares_y),self.square_size, marker_size, self.a_dict)
        if legacy:
            self.board.setLegacyPattern(True)
        self.point_data = np.asarray(self.board.getChessboardCorners(), dtype=np.float64).squeeze()

        self.detection_options = detection_options or {}
        self.detection_params = self._build_charuco_parameters(self.detection_options)
        self.detector_params = self._build_detector_parameters(self.detection_options)
        self.refine_params = self._build_refine_parameters(self.detection_options)
        try:
            self.board_detectors = aruco.CharucoDetector(
                self.board,
                self.detection_params,
                self.detector_params,
                self.refine_params,
            )
        except TypeError:
            # Backward compatibility with OpenCV builds that only expose
            # CharucoDetector(board, charucoParams).
            logging.warning(
                "OpenCV CharucoDetector constructor does not support DetectorParameters/RefineParameters; "
                "falling back to CharucoParameters-only detector construction."
            )
            self.board_detectors = aruco.CharucoDetector(self.board, self.detection_params)
        self.given_legacy_warning = False

        self._process_data()

    @staticmethod
    def _coerce_corner_refinement_method(value):
        if isinstance(value, str):
            return getattr(aruco, value, value)
        return value

    @classmethod
    def _build_charuco_parameters(cls, detection_options: dict) -> aruco.CharucoParameters:
        params = aruco.CharucoParameters()
        params.tryRefineMarkers = True
        for key, value in detection_options.get("CharucoParameters", {}).items():
            if not hasattr(params, key):
                continue
            if value is None:
                continue
            if key in {"cameraMatrix", "distCoeffs"}:
                # These fields must be numpy arrays for OpenCV's C++ bindings.
                value = np.asarray(value, dtype=np.float64)
            setattr(params, key, value)
        return params

    @classmethod
    def _build_detector_parameters(cls, detection_options: dict) -> aruco.DetectorParameters:
        params = aruco.DetectorParameters()
        for key, value in detection_options.get("DetectorParameters", {}).items():
            if not hasattr(params, key):
                continue
            if value is None:
                continue
            if key == "cornerRefinementMethod":
                value = cls._coerce_corner_refinement_method(value)
            setattr(params, key, value)
        return params

    @staticmethod
    def _build_refine_parameters(detection_options: dict) -> aruco.RefineParameters:
        params = aruco.RefineParameters()
        for key, value in detection_options.get("RefineParameters", {}).items():
            if not hasattr(params, key):
                continue
            if value is None:
                continue
            setattr(params, key, value)
        return params

    def _board_size_mm(self) -> tuple[float, float]:
        n_x, n_y = self.board.getChessboardSize()
        side_mm = float(self.square_size) * 1000.0
        return float(n_x) * side_mm, float(n_y) * side_mm

    def _render_board(self, px_per_mm: float = 12.0) -> np.ndarray:
        width_mm, height_mm = self._board_size_mm()
        width_px = max(1, int(round(width_mm * px_per_mm)))
        height_px = max(1, int(round(height_mm * px_per_mm)))
        return self.board.generateImage((width_px, height_px))

    def save_to_pdf(
            self,
            f_out: Path | str | None = None,
            data_format: str = "raster",
            dpi: int = 300,
    ) -> Path:
        if f_out is None:
            f_out = Path(
                f"charuco_{self.board.getChessboardSize()[0]}x{self.board.getChessboardSize()[1]}_"
                f"square_{self.square_size * 1000:.2f}mm.pdf"
            )
        else:
            f_out = Path(f_out)

        f_out = f_out.expanduser().with_suffix(".pdf").resolve()
        f_out.parent.mkdir(parents=True, exist_ok=True)

        if data_format == "vector":
            svg_out = f_out.with_suffix(".svg")
            self.save_to_svg(svg_out, suppress_svg_log=True)
            cairosvg.svg2pdf(url=str(svg_out), write_to=str(f_out))
            logging.info("Saved ChArUco Vector PDF: %s", f_out)
            return f_out

        if data_format != "raster":
            raise ValueError("data_format must be one of: raster, vector")

        image = self._render_board()
        with Image.fromarray(image) as im:
            im.save(fp=f_out, resolution=float(dpi))

        logging.info("Saved ChArUco Raster PDF: %s", f_out)
        return f_out

    @staticmethod
    def _bilinear_quad(q: np.ndarray, u: float, v: float) -> np.ndarray:
        # q order: tl, tr, br, bl
        p00 = q[0]
        p10 = q[1]
        p11 = q[2]
        p01 = q[3]
        return (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + u * v * p11 + (1 - u) * v * p01

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

    def iter_marker_slots(self):
        ids = np.asarray(self.board.getIds()).reshape(-1).astype(int)
        obj_points = self.board.getObjPoints()

        if len(ids) != len(obj_points):
            raise ValueError(f"Mismatch between ids ({len(ids)}) and obj_points ({len(obj_points)}) in Charuco board.")

        for marker_id, corners in zip(ids, obj_points):
            c = np.asarray(corners, dtype=float)
            if c.ndim == 3:
                c = c.reshape(-1, c.shape[-1])

            if c.shape[0] != 4 or c.shape[1] < 2:
                raise ValueError(f"Unexpected marker corner shape: {c.shape}")

            yield int(marker_id), c[:, :2]

    def save_to_svg(
            self,
            f_out: Path | str | None = None,
            border_width: float = 10,
            suppress_svg_log: bool = False,
    ) -> Path:
        if f_out is None:
            n_x, n_y = self.board.getChessboardSize()
            f_out = Path(
                f"charuco_{n_x}x{n_y}_square_{self.square_size * 1000:.2f}mm_true_vector.svg"
            )
        else:
            f_out = Path(f_out)

        f_out = f_out.expanduser().with_suffix(".svg").resolve()
        f_out.parent.mkdir(parents=True, exist_ok=True)

        n_cols, n_rows = self.board.getChessboardSize()
        n_cols = int(n_cols)
        n_rows = int(n_rows)
        sq = float(self.square_size)
        black_polys: list[np.ndarray] = []

        # Draw black chessboard squares.
        for r in range(n_rows):
            for c in range(n_cols):
                if (r + c) % 2 != 0:
                    continue
                x0 = c * sq
                y0 = r * sq
                black_polys.append(
                    np.array(
                        [[x0, y0], [x0 + sq, y0], [x0 + sq, y0 + sq], [x0, y0 + sq]],
                        dtype=float,
                    )
                )

        dct = self.board.getDictionary()
        for marker_id, q in self.iter_marker_slots():
            grid = self.aruco_marker_grid_for_id(dct, marker_id)
            g_rows, g_cols = grid.shape

            for r in range(g_rows):
                v0 = r / g_rows
                v1 = (r + 1) / g_rows
                for c in range(g_cols):
                    if grid[r, c] != 1:
                        continue
                    u0 = c / g_cols
                    u1 = (c + 1) / g_cols
                    black_polys.append(
                        np.array(
                            [
                                self._bilinear_quad(q, u0, v0),
                                self._bilinear_quad(q, u1, v0),
                                self._bilinear_quad(q, u1, v1),
                                self._bilinear_quad(q, u0, v1),
                            ],
                            dtype=float,
                        )
                    )

        if not black_polys:
            raise RuntimeError("No vector marker polygons were generated for SVG export.")

        all_pts = np.vstack(black_polys)
        min_xy = all_pts.min(axis=0)
        max_xy = all_pts.max(axis=0)

        border_m = float(border_width) * 0.001
        min_xy -= border_m
        max_xy += border_m
        canvas_w = float(max_xy[0] - min_xy[0])
        canvas_h = float(max_xy[1] - min_xy[1])

        dwg = svgwrite.Drawing(
            str(f_out),
            size=(f"{canvas_w * 1000.0:.6f}mm", f"{canvas_h * 1000.0:.6f}mm"),
            viewBox=f"0 0 {canvas_w:.6f} {canvas_h:.6f}",
        )
        dwg.add(dwg.rect(insert=(0, 0), size=(canvas_w, canvas_h), fill="white"))

        offset = -min_xy
        for poly in black_polys:
            p = poly + offset
            dwg.add(dwg.polygon(points=[tuple(xy) for xy in p], fill="black", stroke="none"))

        svg_text = dwg.tostring()
        with open(f_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg_text)
            fh.flush()
            import os
            os.fsync(fh.fileno())

        if (not f_out.exists()) or f_out.stat().st_size == 0:
            raise IOError(f"SVG write failed: {f_out}")

        if not suppress_svg_log:
            logging.info("Saved ChArUco SVG: %s", f_out)
        return f_out


    def find_in_image(self, image, draw=False, camera: Camera|None = None, wait_len=1) -> ImageDetection:
        """
        Detects features of this target in the input image.

        :param image: The image to detect in.
        :param draw: Whether or not the detected corners should be drawn.
        :param camera: optional. A camera target for more accurate detection.
        :param wait_len: time to pause to allow drawing of detections. -1 waits for key press.

        :return ImageDetection: a data class wrapping the data detected in the image.
        """
        # c_corners, c_ids, od = adaptive_decimated_charuco_detection_stereo(image, charuco_board=self.board, aruco_dict=self.a_dict)
        # _, _, mloc, mid = self.board_detectors.detectBoard(image)
        c_corners, c_ids, mloc, mid = self.board_detectors.detectBoard(image) #, markerCorners=mloc, markerIds=mid)
        if c_corners is None and mloc is not None:
            if not self.given_legacy_warning:
                logging.warning("Found markers, but no corners, trying using alternative board detection")
                self.given_legacy_warning = True
            am_legacy = self.board.getLegacyPattern()
            self.board.setLegacyPattern(not am_legacy)
            c_corners, c_ids, mloc, mid = self.board_detectors.detectBoard(image, markerCorners=mloc, markerIds=mid)

        od = 1

        if c_corners is None:
            return ImageDetection() # return an empty detection

            # aruco.drawDetectedMarkers(display_im, np.array(corners)/d_f, ids)

        if draw:
            display_im = image.copy()
            target_size = [480, 640]
            d_f = int(max((min(np.array(display_im.shape[:2]) / target_size)), 1))
            display_im = downsample_valid(display_im[:,:,0] if display_im.ndim > 2 else display_im, d_f).astype(np.uint8)
            # d_f=1
            if display_im.ndim == 2:
                display_im = np.tile(display_im[..., None], (1, 1, 3))
            aruco.drawDetectedCornersCharuco(
                display_im,
                np.array(c_corners) / d_f,
                c_ids,
            )

            cv2.imshow('detections', display_im)
            cv2.waitKey(wait_len)


        return ImageDetection(c_ids[:, 0], c_corners[:, 0])

    def plot(self,imres=(1000,1000)):
        """
        Draws the target as a matplotlib plot.
        """
        plt.imshow(self.board.generateImage(imres), cmap='gray')
        plt.show()


if __name__ == '__main__':
    test = ChArUco(num_squares_x=7, num_squares_y=7, square_size=4)
    test.plot()
    # test.get_printable_texture()
