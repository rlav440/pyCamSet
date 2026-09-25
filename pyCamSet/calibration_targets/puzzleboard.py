from __future__ import annotations

from io import BytesIO
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import svgwrite

from pyCamSet.calibration_targets.core import AbstractTarget, ImageDetection
from pyCamSet.calibration_targets.core.abstract_target import (
    EXPORT_SUFFIXES, export_path)
from pyCamSet.calibration_targets.core.parameters import (
    DocumentedParameters,
    Parameterisation,
)
from pyCamSet.calibration_targets.markers.puzzleboard import (
    PUZZLEBOARD_DETECTOR,
    detect_puzzleboard_image,
)
from pyCamSet.cameras import Camera


_BASE_CODE = (
    "00001011100001110101010100100011010110110000111011100010100100010001111110000010101111110100110010010101110101110000111111011011010110011011111011100111101001111010001\n"
    "01100001111110111111110100010010000011110001001100000100010001100001100000011100011010100011101110101101110110010010000110100110001110101000111000100110001111010100100\n"
    "01000001010000111010000111101011111010000010010010000101110100110110011101011010110010110010010010110011001011110111001110000000101011011111010110110011000011110100011"
)
_BASE_CODE2 = (
    "11111010100111111000010100110111100100000100110010011111100011010001110110100100001000111001111010000000100111000000010111110010110101111101110100101101011001001110010\n"
    "10110011000100110101100111011000011010100010010101011000001011000000111000001011011000011111010101101101100011001111001001001010111011011001110011001011100011100011001\n"
    "00110000010101000111101010010110000110010010000000110000111101000111101100101010100100101001011111101000000010110101101011111110001000111101011101100101000011011111111"
)

_CODE_ROWS = tuple(_BASE_CODE.splitlines())
_CODE_ROWS2 = tuple(_BASE_CODE2.splitlines())
_CODE_HEIGHT = len(_CODE_ROWS)
_CODE_WIDTH = len(_CODE_ROWS[0])
_CODE_SIZE = _CODE_HEIGHT * _CODE_WIDTH


if len(_CODE_ROWS) != len(_CODE_ROWS2) or any(len(a) != len(b) for a, b in zip(_CODE_ROWS, _CODE_ROWS2)):
    raise ValueError("PuzzleBoard base-code dimensions do not match.")


def _generate_code() -> np.ndarray:
    """Create the combined periodic code field used by the JavaScript generator."""
    code = np.zeros((_CODE_SIZE, _CODE_SIZE), dtype=np.uint8)
    for y in range(_CODE_SIZE):
        for x in range(_CODE_SIZE):
            a = int(_CODE_ROWS[y % _CODE_HEIGHT][x % _CODE_WIDTH] == "1")
            b = 2 * int(_CODE_ROWS2[x % _CODE_HEIGHT][y % _CODE_WIDTH] == "1")  # bit 2, transposed
            code[y, x] = a + b
    return code


_CODE_FIELD = _generate_code()


class PuzzleBoard(AbstractTarget):
    """Define a PuzzleBoard target, detector adapter, and vector export methods."""

    DETECTOR_BACKENDS = {"puzzle_board": PUZZLEBOARD_DETECTOR}

    @classmethod
    def construction_parameters(cls, backend: str | None = None) -> Parameterisation:
        """What decides where a board's corners are, and how it prints."""
        return DocumentedParameters(
            cls.__init__,
            "num_squares_x", "num_squares_y", "square_size",
            "start_x", "start_y", "paper_width", "paper_height")

    @classmethod
    def export_parameters(cls) -> Parameterisation:
        """How a PuzzleBoard is drawn, which is not what it is."""
        return DocumentedParameters(cls.save_printable, "dpi")

    @classmethod
    def printable_name(cls, values: dict, kind: str = "svg") -> str:
        return (f"puzzleboard_{int(values['num_squares_x'])}x"
                f"{int(values['num_squares_y'])}_"
                f"{float(values['square_size']):g}mm{EXPORT_SUFFIXES[kind]}")

    def save_printable(self, path, kind: str = "svg", dpi: int = 300) -> Path:
        """
        Write this board as a file to print.

        :param path: where to write it
        :param kind: one of :data:`EXPORT_KINDS`
        :param dpi: Raster DPI -- the resolution a raster PDF is rendered
            at. Ignored by the vector formats. Suggested: 300-600.
        :raises ValueError: for a format a board cannot be written as
        """
        if kind == "svg":
            return self.save_to_svg(path)
        if kind == "pdf_vector":
            return self.save_to_pdf(path, data_format="vector")
        if kind == "pdf_raster":
            return self.save_to_pdf(path, data_format="raster", dpi=int(dpi))
        raise ValueError(f"A PuzzleBoard cannot be written as {kind!r}.")

    def __init__(
        self,
        num_squares_x: int = 105,
        num_squares_y: int = 148,
        square_size: float = 2.0,
        start_x: int = 0,
        start_y: int = 0,
        paper_width: float = 210.0,
        paper_height: float = 297.0,
        detection_options: dict | None = None,
    ):
        """
        Initialise a PuzzleBoard target with dimensions in millimetres.

        :param num_squares_x: Corners across -- corners along the printed
            board's x axis. Suggested: fills the page at the chosen square
            size.
        :param num_squares_y: Corners down -- corners along the printed
            board's y axis. Suggested: fills the page at the chosen square
            size.
        :param square_size: Square size (mm) -- the printed edge length of
            one square, in millimetres. Suggested: 2.
        :param start_x: Code origin x -- where in the periodic code this
            printed window begins. Two boards cut from different windows
            decode to different keys. Suggested: 0.
        :param start_y: Code origin y -- where in the periodic code this
            printed window begins, down the page. Suggested: 0.
        :param paper_width: Page width (mm) -- the page the board is
            centred on. Suggested: 210 (A4).
        :param paper_height: Page height (mm) -- the page the board is
            centred on. Suggested: 297 (A4).
        :param detection_options: what the detector is told, by the keys
            :meth:`detector_parameterisation` describes.
        """
        super().__init__(inputs=locals())
        self.num_squares_x = int(num_squares_x)
        self.num_squares_y = int(num_squares_y)
        self.square_size = float(square_size)
        self.start_x = int(start_x)
        self.start_y = int(start_y)
        self.paper_width = float(paper_width)
        self.paper_height = float(paper_height)
        self._validate_dimensions()
        self.point_data = self._make_point_data()
        self._process_data()

    def _validate_dimensions(self) -> None:
        """Validate the finite periodic-code window and physical dimensions."""
        if self.num_squares_x < 2 or self.num_squares_y < 2:
            raise ValueError("num_squares_x and num_squares_y must both be at least 2.")
        if self.square_size <= 0:
            raise ValueError("square_size must be greater than zero.")
        if self.paper_width <= 0 or self.paper_height <= 0:
            raise ValueError("paper_width and paper_height must be greater than zero.")
        if self.start_x < 0 or self.start_y < 0:
            raise ValueError("start_x and start_y must be non-negative.")
        if self.start_x + self.num_squares_x > _CODE_SIZE:
            raise ValueError(f"start_x + num_squares_x must not exceed {_CODE_SIZE}.")
        if self.start_y + self.num_squares_y > _CODE_SIZE:
            raise ValueError(f"start_y + num_squares_y must not exceed {_CODE_SIZE}.")

    def _make_point_data(self) -> np.ndarray:
        """Create flat pyCamSet object points indexed by global PuzzleBoard coordinates."""
        points = np.zeros((1, _CODE_SIZE * _CODE_SIZE, 3), dtype=np.float64)
        for row in range(_CODE_SIZE):
            for col in range(_CODE_SIZE):
                point_id = row * _CODE_SIZE + col
                # Half a square, because the feature the detector reports for
                # code (row, col) is the corner where four squares meet, and the
                # squares are centred on the integer code positions.
                points[0, point_id, 0] = (col - self.start_x + 0.5) * self.square_size
                points[0, point_id, 1] = (row - self.start_y + 0.5) * self.square_size
        return points

    def _board_offsets(self) -> tuple[float, float]:
        """Return the centred SVG offset in millimetres."""
        off_x = (self.paper_width - self.num_squares_x * self.square_size) / 2.0
        off_y = (self.paper_height - self.num_squares_y * self.square_size) / 2.0
        return off_x, off_y

    def _svg_document(self) -> svgwrite.Drawing:
        """Construct the vector target using the exact loops from puzzle.js."""
        off_x, off_y = self._board_offsets()
        drawing = svgwrite.Drawing(
            size=(f"{self.paper_width:g}mm", f"{self.paper_height:g}mm"),
            viewBox=f"0 0 {self.paper_width:g} {self.paper_height:g}",
        )
        drawing.add(drawing.rect(insert=(0, 0), size=(self.paper_width, self.paper_height), fill="white"))

        for x in range(self.num_squares_x + 1):
            for y in range(self.num_squares_y + 1):
                if (x + y + self.start_x + self.start_y) % 2 == 1:
                    continue
                xx = x - 0.5
                yy = y - 0.5
                width = 1.0
                height = 1.0
                if x == 0:
                    xx += 0.5
                    width = 0.5
                if y == 0:
                    yy += 0.5
                    height = 0.5
                if x == self.num_squares_x:
                    width = 0.5
                if y == self.num_squares_y:
                    height = 0.5
                drawing.add(drawing.rect(
                    insert=(off_x + xx * self.square_size, off_y + yy * self.square_size),
                    size=(width * self.square_size, height * self.square_size),
                    fill="black",
                    stroke="none",
                ))

        radius = self.square_size / 6.0
        for y in range(self.num_squares_y):
            for x in range(self.num_squares_x - 1):
                value = int(_CODE_FIELD[self.start_y + y, self.start_x + x])
                fill = "white" if value & 2 else "black"  # bit 2 is the horizontal edge
                drawing.add(drawing.circle(
                    center=(off_x + (1 + x) * self.square_size, off_y + (0.5 + y) * self.square_size),
                    r=radius,
                    fill=fill,
                    stroke="none",
                ))

        for y in range(self.num_squares_y - 1):
            for x in range(self.num_squares_x):
                value = int(_CODE_FIELD[self.start_y + y, self.start_x + x])
                fill = "white" if value & 1 else "black"  # bit 1 is the vertical edge
                drawing.add(drawing.circle(
                    center=(off_x + (0.5 + x) * self.square_size, off_y + (1 + y) * self.square_size),
                    r=radius,
                    fill=fill,
                    stroke="none",
                ))
        return drawing

    def save_to_svg(
        self,
        f_out: Path | str,
        suppress_svg_log: bool = False,
    ) -> Path:
        """Save the PuzzleBoard as a physically sized vector SVG."""
        f_out = export_path(f_out, ".svg")
        svg_text = self._svg_document().tostring()
        with open(f_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg_text)
            fh.flush()
        if (not f_out.exists()) or f_out.stat().st_size == 0:
            raise IOError(f"SVG write failed: {f_out}")
        if not suppress_svg_log:
            logging.info("Saved PuzzleBoard SVG: %s", f_out)
        return f_out

    def save_to_pdf(
        self,
        f_out: Path | str,
        data_format: str = "raster",
        dpi: int = 300,
    ) -> Path:
        """Save a raster or vector PDF using the same API as the ChArUco target."""
        f_out = export_path(f_out, ".pdf")
        svg_text = self._svg_document().tostring()
        from pyCamSet.utils.cairo_dll_helper import cairosvg_or_explain
        cairosvg = cairosvg_or_explain()
        if data_format == "vector":
            cairosvg.svg2pdf(bytestring=svg_text.encode("utf-8"), write_to=str(f_out))
            logging.info("Saved PuzzleBoard Vector PDF: %s", f_out)
            return f_out
        if data_format != "raster":
            raise ValueError("data_format must be one of: raster, vector")
        png = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=max(1, int(round(self.paper_width / 25.4 * dpi))),
            output_height=max(1, int(round(self.paper_height / 25.4 * dpi))),
        )
        with Image.open(BytesIO(png)) as image:
            image.convert("RGB").save(f_out, resolution=float(dpi))
        logging.info("Saved PuzzleBoard Raster PDF: %s", f_out)
        return f_out

    def find_in_image(
        self,
        image,
        draw: bool = False,
        camera: Camera | None = None,
        wait_len: int = 1,
    ) -> ImageDetection:
        """Detect PuzzleBoard grid points and return pyCamSet-compatible IDs."""
        del camera
        point_ids, point_coords = detect_puzzleboard_image(image, min_width=self.detection_options["min_width"])
        if not point_ids:
            return ImageDetection()
        positions = np.asarray(point_ids, dtype=np.int64)
        coordinates = np.asarray(point_coords, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("PuzzleBoard detector positions must have shape (n, 2).")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("PuzzleBoard detector coordinates must have shape (n, 2).")
        valid = (
            (positions[:, 0] >= 0)
            & (positions[:, 0] < _CODE_SIZE)
            & (positions[:, 1] >= 0)
            & (positions[:, 1] < _CODE_SIZE)
        )
        positions = positions[valid]
        coordinates = coordinates[valid]
        keys = positions[:, 0] * _CODE_SIZE + positions[:, 1]
        image_points = coordinates[:, ::-1]  # the detector reports [row, column]
        if draw:
            display_im = np.asarray(image).copy()
            for x, y in image_points:
                cv2.circle(display_im, (int(round(x)), int(round(y))), 3, (0, 0, 255), 1)
            cv2.imshow("PuzzleBoard detections", display_im)
            cv2.waitKey(wait_len)
        return ImageDetection(keys.astype(np.int64), image_points)

    def plot(self, imres: tuple[int, int] = (1000, 1000)) -> None:
        """Display a rasterised preview of the vector target."""
        from pyCamSet.utils.cairo_dll_helper import cairosvg_or_explain
        cairosvg = cairosvg_or_explain()
        # Preview the board, not the page it is centred on.  A board smaller
        # than its paper is a speck in the middle of a blank sheet otherwise,
        # and the aspect has to be kept: a preview whose squares are not square
        # is the one thing this preview is looked at for.
        margin = self.square_size  # one square of surrounding page, so the board reads as a board
        view_width = self.num_squares_x * self.square_size + 2 * margin
        view_height = self.num_squares_y * self.square_size + 2 * margin
        fit = min(imres[0] / view_width, imres[1] / view_height)  # pixels per millimetre
        png = cairosvg.svg2png(
            bytestring=self._svg_document().tostring().encode("utf-8"),
            output_width=int(self.paper_width * fit),
            output_height=int(self.paper_height * fit),
        )
        off_x, off_y = self._board_offsets()
        with Image.open(BytesIO(png)) as image:
            left = max(0, int((off_x - margin) * fit))  # clamped, so a board filling its page is not padded
            top = max(0, int((off_y - margin) * fit))
            view = image.crop((
                left, top,
                min(image.width, left + int(view_width * fit)),
                min(image.height, top + int(view_height * fit)),
            ))
            plt.imshow(np.asarray(view), cmap="gray")
        plt.axis("off")
        plt.show()

