from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import cv2
import numpy as np
import svgwrite
from PIL import Image

logger = logging.getLogger(__name__)

from pyCamSet.calibration_targets.core.abstract_target import (
    AbstractTarget, EXPORT_SUFFIXES,
)
from pyCamSet.calibration_targets.core.parameters import (
    DocumentedParameters,
    Parameterisation,
)
from pyCamSet.calibration_targets.core.target_detections import ImageDetection
from pyCamSet.calibration_targets.markers.backend_registry import (
    ARUCO2_BACKEND,
    dict_names_for_backend,
    dictionary_id,
)
from pyCamSet.calibration_targets.markers.aruco2_gridboard import (
    CHARUCO2_DETECTOR,
    detect_grid_board_corners,
    dictionary_marker_bits,
    render_grid_board_image,
)
from pyCamSet.cameras import Camera

#: The smallest board aruco2's own layout maths tolerates -- a 1x1 grid board
#: has no ambiguity in its corner ids, but "at least 2 markers each way" is
#: the smallest shape a person would call a *board* rather than a marker.
_MIN_SQUARES = 2

#: The dictionary a board is printed with unless another is asked for --
#: same default as ChArUco, for the same reason (a large, high-id-count
#: dictionary that will not run out of markers on a big board).
_DEFAULT_DICT_NAME = "DICT_4X4_1000"

#: aruco2's grid board places one marker per square, so a dictionary too
#: sparse to cover a board is a real hazard, unlike ChArUco/Ccube's sparser
#: placement -- but AprilTag dictionaries are also excluded here for a
#: second reason: they are OpenCV's own predefined tag families, not
#: aruco2's, and are not validated against aruco2's grid-board detector.
_EXCLUDED_DICT_PREFIXES = ("DICT_APRILTAG_",)


class ChArUco2(AbstractTarget):
    """
    A planar ChArUco2 board: aruco2's ``GridBoard`` design.

    Unlike :class:`~pyCamSet.calibration_targets.charuco.target.ChArUco`,
    which places a sparse marker only on every other square, ChArUco2 places
    an ArUco marker on *every* square -- a standard marker on a black
    square, an inverted one on a white square. The design is described in
    https://www.sciencedirect.com/science/article/pii/S2352711026003249.
    An N x M board (in markers) has (N+1) x (M+1) observable intersection
    corners, including the board's own border, and it has no OpenCV
    (``aruco1``) equivalent -- OpenCV's ``cv2.aruco`` module has no
    every-square board design under any name -- so this target is read with
    aruco2 only; there is no backend to choose.

    **Detection is a fixed pipeline with nothing to tune.**
    ``aruco2.detect_grid_board`` takes an image, a board size and a
    dictionary and nothing else -- there is no ``DetectionParameters``-style
    control for a form to show or a study to sweep, unlike
    :class:`ChArUco`/:class:`~pyCamSet.calibration_targets.ccube.target
    .Ccube`'s OpenCV-backed detection.

    **ChArUco2 has not been validated on a real printed and photographed
    board.** Every check behind it -- the corner-id mapping, occlusion,
    rotation and perspective-warp recovery, and the SVG round trip -- runs
    against aruco2's own rendered raster, not against a camera photograph
    of a printed board. Treat detection quality on a real capture as
    unverified until it has been.
    """

    DETECTOR_BACKENDS = {ARUCO2_BACKEND: CHARUCO2_DETECTOR}

    @classmethod
    def construction_parameters(cls, backend: str | None = None) -> Parameterisation:
        """What decides where a board's corners are, and how it prints."""
        dict_names = [
            name for name in dict_names_for_backend(ARUCO2_BACKEND)
            if not name.startswith(_EXCLUDED_DICT_PREFIXES)
        ]
        return DocumentedParameters(
            cls.__init__,
            "num_squares_x", "num_squares_y", "square_size", "a_dict",
            choices={"a_dict": dict_names},
        )

    @classmethod
    def export_parameters(cls) -> Parameterisation:
        """How a ChArUco2 board is drawn, which is not what it is."""
        return DocumentedParameters(cls.save_printable, "border_width", "dpi")

    @classmethod
    def printable_name(cls, values: dict, kind: str = "svg") -> str:
        return (f"charuco2_{int(values['num_squares_x'])}x"
                f"{int(values['num_squares_y'])}_"
                f"{float(values['square_size']):g}mm{EXPORT_SUFFIXES[kind]}")

    def __init__(
        self,
        num_squares_x: int = 5,
        num_squares_y: int = 5,
        square_size: float = 10.0,
        a_dict=_DEFAULT_DICT_NAME,
        detection_options: dict | None = None,
    ):
        """
        Initialises a ChArUco2 board in mm.

        :param num_squares_x: Squares across -- marker squares along the
            board's x axis. Suggested: 5-20.
        :param num_squares_y: Squares down -- marker squares along the
            board's y axis. Suggested: 5-20.
        :param square_size: Square size (mm) -- the printed edge length of
            one marker square, in millimetres. Suggested: 4-40.
        :param a_dict: ArUco dictionary -- the marker alphabet printed on
            the board. It must have at least one marker per square.
        :param detection_options: what the detector is told, by the keys
            :meth:`detector_parameterisation` describes. aruco2's grid-board
            detector takes none, so this is always empty for ChArUco2.
        """
        super().__init__(inputs=locals(), backend=ARUCO2_BACKEND)

        if num_squares_x < _MIN_SQUARES or num_squares_y < _MIN_SQUARES:
            raise ValueError(
                f"A ChArUco2 board must be at least {_MIN_SQUARES}x"
                f"{_MIN_SQUARES} squares; got {num_squares_x}x{num_squares_y}.")
        if square_size <= 0:
            raise ValueError(
                f"A marker square is printed, so it has a size; got "
                f"square_size={square_size}.")

        self.num_squares_x = int(num_squares_x)
        self.num_squares_y = int(num_squares_y)
        self.grid_size = (self.num_squares_x, self.num_squares_y)
        self.square_size = float(square_size) / 1000.0  # metres
        self._aruco_dict_int = dictionary_id(a_dict, ARUCO2_BACKEND)

        # (grid_x+1) x (grid_y+1) intersection corners, row-major, so a
        # corner's array index equals aruco2's own global corner id (gid =
        # row * (grid_x+1) + col; verified empirically, by round-tripping
        # detection against a full, an occluded, a rotated and a
        # perspective-warped board -- see aruco2_gridboard.detect_grid_board_corners).
        cols, rows = np.meshgrid(
            np.arange(self.num_squares_x + 1, dtype=np.float64),
            np.arange(self.num_squares_y + 1, dtype=np.float64),
        )
        self.point_data = np.stack(
            [cols.ravel() * self.square_size,
             rows.ravel() * self.square_size,
             np.zeros(cols.size, dtype=np.float64)],
            axis=-1,
        )

        self._process_data()

    # -- detection --------------------------------------------------------

    def find_in_image(self, image, draw=False, camera: Camera | None = None,
                       wait_len=1) -> ImageDetection:
        """
        Detects this board in an image with aruco2's grid-board detector.

        :param image: The image to detect in.
        :param draw: Whether or not the detected corners should be drawn.
        :param camera: unused -- aruco2's grid-board detector does not take
            a camera model.
        :param wait_len: time to pause to allow drawing of detections. -1
            waits for key press.
        :return ImageDetection: the corners found, keyed by the row-major
            index :attr:`point_data` was built with.
        """
        corner_ids, image_points = detect_grid_board_corners(
            image, self.grid_size, self._aruco_dict_int, self.square_size)
        if corner_ids is None:
            return ImageDetection()

        # point_data gains a leading "face" axis once built (a single face
        # here, but AbstractTarget.make_local() wraps any 2D point_data into
        # (1, n, 3)) -- the corner count is the second-to-last axis, not the
        # first.
        n_corners = self.point_data.shape[-2]
        valid = (corner_ids >= 0) & (corner_ids < n_corners)
        if not np.all(valid):
            logger.warning(
                "ChArUco2: dropping %d detected corner(s) outside the "
                "board's %d known corners.",
                int((~valid).sum()), n_corners)
            corner_ids = corner_ids[valid]
            image_points = image_points[valid]

        if draw:
            display_im = image.copy() if hasattr(image, "copy") else np.asarray(image).copy()
            if display_im.ndim == 2:
                display_im = np.tile(display_im[..., None], (1, 1, 3))
            for pt in image_points:
                cv2.circle(display_im, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            cv2.imshow('detections', display_im)
            cv2.waitKey(wait_len)

        return ImageDetection(corner_ids.astype(np.int64), image_points)

    # -- printing -----------------------------------------------------------

    def _marker_bits(self) -> int:
        return dictionary_marker_bits(self._aruco_dict_int)

    def _render(self, dpi: float) -> tuple[np.ndarray, float]:
        """
        The board raster aruco2 would both print and detect, at *dpi*.

        Unlike ``ChArUco._render_board``, which hardcodes 12 px/mm no
        matter what dpi was asked for, ``bit_size`` here is chosen so the
        rendered image's actual pixels-per-mm matches ``dpi / 25.4`` as
        closely as an integer bit size allows.

        :return: ``(image, px_per_mm)`` -- the image, and the px/mm it was
            actually rendered at (nearest integer ``bit_size`` to the
            request, so not always exactly ``dpi / 25.4``).
        """
        marker_bits = self._marker_bits()
        square_size_mm = self.square_size * 1000.0
        wanted_px_per_mm = float(dpi) / 25.4
        bit_size = max(1, round(wanted_px_per_mm * square_size_mm / marker_bits))
        image = render_grid_board_image(self.grid_size, self._aruco_dict_int, bit_size)
        actual_px_per_mm = (marker_bits * bit_size) / square_size_mm
        return image, actual_px_per_mm

    def save_printable(self, path, kind: str = "svg", border_width: float = 10.0,
                        dpi: int = 300) -> Path:
        """
        Write this board as a file to print.

        :param path: where to write it
        :param kind: one of :data:`EXPORT_KINDS`
        :param border_width: Border (mm) -- the white margin drawn around
            the board, in millimetres, on top of the marker border aruco2
            already draws into the image itself. Detection: markers at the
            very edge of a page are harder to find. Suggested: 10.
        :param dpi: Raster DPI -- the resolution the board is rendered at.
            Every export kind is a raster of aruco2's own board image (the
            same image aruco2 both prints and detects from), sized so its
            actual pixels-per-mm matches this value. Suggested: 300-600.
        :raises ValueError: for a format a board cannot be written as
        """
        if kind == "svg":
            return self.save_to_svg(path, border_width=border_width, dpi=dpi)
        if kind == "pdf_vector":
            return self.save_to_pdf(path, data_format="vector",
                                     border_width=border_width, dpi=dpi)
        if kind == "pdf_raster":
            return self.save_to_pdf(path, data_format="raster",
                                     border_width=border_width, dpi=int(dpi))
        raise ValueError(f"A ChArUco2 cannot be written as {kind!r}.")

    def save_to_svg(
            self,
            f_out: Path | str | None = None,
            border_width: float = 10.0,
            dpi: float = 300.0,
            suppress_svg_log: bool = False,
    ) -> Path:
        """Embed aruco2's own board raster in an SVG, at true mm scale."""
        if f_out is None:
            f_out = Path(
                f"charuco2_{self.num_squares_x}x{self.num_squares_y}_"
                f"square_{self.square_size * 1000:.2f}mm.svg"
            )
        else:
            f_out = Path(f_out)
        f_out = f_out.expanduser().with_suffix(".svg").resolve()
        f_out.parent.mkdir(parents=True, exist_ok=True)

        image, px_per_mm = self._render(dpi)
        board_w_mm = image.shape[1] / px_per_mm
        board_h_mm = image.shape[0] / px_per_mm
        canvas_w = board_w_mm + 2 * float(border_width)
        canvas_h = board_h_mm + 2 * float(border_width)

        png_bytes = io.BytesIO()
        Image.fromarray(image).save(png_bytes, format="PNG")
        data_uri = "data:image/png;base64," + base64.b64encode(
            png_bytes.getvalue()).decode("ascii")

        dwg = svgwrite.Drawing(
            str(f_out),
            size=(f"{canvas_w:.6f}mm", f"{canvas_h:.6f}mm"),
            viewBox=f"0 0 {canvas_w:.6f} {canvas_h:.6f}",
        )
        dwg.add(dwg.rect(insert=(0, 0), size=(canvas_w, canvas_h), fill="white"))
        dwg.add(dwg.image(
            href=data_uri,
            insert=(float(border_width), float(border_width)),
            size=(board_w_mm, board_h_mm),
        ))

        svg_text = dwg.tostring()
        with open(f_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg_text)
            fh.flush()
            import os
            os.fsync(fh.fileno())

        if (not f_out.exists()) or f_out.stat().st_size == 0:
            raise IOError(f"SVG write failed: {f_out}")

        if not suppress_svg_log:
            logger.info("Saved ChArUco2 SVG: %s", f_out)
        return f_out

    def save_to_pdf(
            self,
            f_out: Path | str | None = None,
            data_format: str = "raster",
            border_width: float = 10.0,
            dpi: float = 300.0,
    ) -> Path:
        if f_out is None:
            f_out = Path(
                f"charuco2_{self.num_squares_x}x{self.num_squares_y}_"
                f"square_{self.square_size * 1000:.2f}mm.pdf"
            )
        else:
            f_out = Path(f_out)
        f_out = f_out.expanduser().with_suffix(".pdf").resolve()
        f_out.parent.mkdir(parents=True, exist_ok=True)

        if data_format == "vector":
            try:
                import pyCamSet.utils.cairo_dll_helper  # noqa: F401
                import cairosvg
            except OSError as _cairo_err:
                raise OSError(
                    f"{_cairo_err}\n\n"
                    "pyCamSet's ChArUco2 target code requires the native "
                    "'cairo' library, which cairosvg requires but pip "
                    "cannot install on its own.\n"
                    "Install the native cairo library for your platform, "
                    "then re-import pyCamSet:\n"
                    "  - conda (Windows/Linux/macOS):  conda install -c conda-forge cairo\n"
                    "  - Debian/Ubuntu:                 apt install libcairo2\n"
                    "  - macOS (Homebrew):              brew install cairo\n"
                    "  - Windows (no conda):            install GTK/cairo and put the DLL on PATH"
                ) from _cairo_err
            svg_out = f_out.with_suffix(".svg")
            self.save_to_svg(svg_out, border_width=border_width, dpi=dpi,
                              suppress_svg_log=True)
            cairosvg.svg2pdf(url=str(svg_out), write_to=str(f_out))
            logger.info("Saved ChArUco2 Vector PDF: %s", f_out)
            return f_out

        if data_format != "raster":
            raise ValueError("data_format must be one of: raster, vector")

        image, px_per_mm = self._render(dpi)
        border_px = int(round(float(border_width) * px_per_mm))
        if border_px > 0:
            padded = np.full(
                (image.shape[0] + 2 * border_px, image.shape[1] + 2 * border_px),
                255, dtype=np.uint8)
            padded[border_px:border_px + image.shape[0],
                   border_px:border_px + image.shape[1]] = image
            image = padded
        resolution_dpi = px_per_mm * 25.4
        with Image.fromarray(image) as im:
            im.save(fp=f_out, resolution=float(resolution_dpi))

        logger.info("Saved ChArUco2 Raster PDF: %s", f_out)
        return f_out

    def plot(self, dpi: float = 150.0):
        """Draws the target as a matplotlib plot."""
        from matplotlib import pyplot as plt
        image, _ = self._render(dpi)
        plt.imshow(image, cmap='gray')
        plt.show()
