'''
Purpose: Define PuzzleBoard calibration targets and vector exports for pyCamSet.
Status: Active target implementation backed by the installed PuzzleBoard detector.
Future: Add target-specific detector tuning once the external package exposes options.
'''
from __future__ import annotations  # Keep postponed annotations consistent with pyCamSet targets.

from io import BytesIO  # Hold temporary SVG rasterisations in memory for PDF/plot output.
import logging  # Report completed target exports.
from pathlib import Path  # Provide the same path handling as the ChArUco target.

import cairosvg  # Convert the vector SVG to PDF or PNG when requested.
import cv2  # Draw optional detections and provide image constants.
import matplotlib.pyplot as plt  # Display the generated target in plot().
import numpy as np  # Store code tables, object points, and detector results.
from PIL import Image  # Convert SVG PNG output to a raster PDF when requested.
import svgwrite  # Write compact SVG primitives directly to disk.

from pyCamSet.calibration_targets import AbstractTarget, ImageDetection  # Reuse pyCamSet target contracts.
from pyCamSet.calibration_targets.puzzleboard_detection import detect_puzzleboard_image  # Use the external detector adapter.
from pyCamSet.cameras import Camera  # Keep the find_in_image signature consistent with other targets.


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

_CODE_ROWS = tuple(_BASE_CODE.splitlines())  # Keep the JavaScript generator's three rows intact.
_CODE_ROWS2 = tuple(_BASE_CODE2.splitlines())  # Keep the second generator code separate for transposition.
_CODE_HEIGHT = len(_CODE_ROWS)  # The base code has three rows.
_CODE_WIDTH = len(_CODE_ROWS[0])  # The expanded periodic code is 501 positions wide.
_CODE_SIZE = _CODE_HEIGHT * _CODE_WIDTH  # The generated square code field is 501 by 501.


if len(_CODE_ROWS) != len(_CODE_ROWS2) or any(len(a) != len(b) for a, b in zip(_CODE_ROWS, _CODE_ROWS2)):
    raise ValueError("PuzzleBoard base-code dimensions do not match.")  # Fail early if a code is edited incorrectly.


def _generate_code() -> np.ndarray:
    """Create the combined periodic code field used by the JavaScript generator."""
    code = np.zeros((_CODE_SIZE, _CODE_SIZE), dtype=np.uint8)  # Allocate the same square field as generateCode().
    for y in range(_CODE_SIZE):  # Repeat each base-code row across the expanded field.
        for x in range(_CODE_SIZE):  # Repeat each base-code column across the expanded field.
            a = int(_CODE_ROWS[y % _CODE_HEIGHT][x % _CODE_WIDTH] == "1")  # Preserve bit 1 from BASE_CODE.
            b = 2 * int(_CODE_ROWS2[x % _CODE_HEIGHT][y % _CODE_WIDTH] == "1")  # Preserve transposed bit 2.
            code[y, x] = a + b  # Store both independent edge bits in one value.
    return code  # Return values in the range 0 through 3.


_CODE_FIELD = _generate_code()  # Build the immutable field once when this module is imported.


class PuzzleBoard(AbstractTarget):
    """Define a PuzzleBoard target, detector adapter, and vector export methods."""

    def __init__(
        self,
        num_squares_x: int = 105,
        num_squares_y: int = 148,
        square_size: float = 2.0,
        start_x: int = 0,
        start_y: int = 0,
        paper_width: float = 210.0,
        paper_height: float = 297.0,
        min_width: int = 4,
        detection_options: dict | None = None,
    ):
        """Initialise a PuzzleBoard target with dimensions expressed in millimetres."""
        super().__init__(inputs=locals())  # Save constructor inputs for pyCamSet multiprocessing and serialisation.
        self.num_squares_x = int(num_squares_x)  # Store the horizontal corner count as an integer.
        self.num_squares_y = int(num_squares_y)  # Store the vertical corner count as an integer.
        self.square_size = float(square_size)  # Store the physical edge length in millimetres.
        self.start_x = int(start_x)  # Store the horizontal code origin used by the printed window.
        self.start_y = int(start_y)  # Store the vertical code origin used by the printed window.
        self.paper_width = float(paper_width)  # Store the SVG page width in millimetres.
        self.paper_height = float(paper_height)  # Store the SVG page height in millimetres.
        self.min_width = int(min_width)  # Store the minimum detector grid width.
        self.detection_options = detection_options or {}  # Retain a compatible extension point for detector settings.
        self._validate_dimensions()  # Reject invalid windows before object-point construction.
        self.point_data = self._make_point_data()  # Index object points by the detector's global grid position.
        self._process_data()  # Compute pyCamSet's local object-point representation.

    def _validate_dimensions(self) -> None:
        """Validate the finite periodic-code window and physical dimensions."""
        if self.num_squares_x < 2 or self.num_squares_y < 2:  # At least one encoded edge is required in each direction.
            raise ValueError("num_squares_x and num_squares_y must both be at least 2.")
        if self.square_size <= 0:  # Physical coordinates cannot use a zero or negative square size.
            raise ValueError("square_size must be greater than zero.")
        if self.paper_width <= 0 or self.paper_height <= 0:  # SVG pages require positive dimensions.
            raise ValueError("paper_width and paper_height must be greater than zero.")
        if self.start_x < 0 or self.start_y < 0:  # The source generator only addresses non-negative code positions.
            raise ValueError("start_x and start_y must be non-negative.")
        if self.start_x + self.num_squares_x > _CODE_SIZE:  # Protect every horizontal and vertical code lookup.
            raise ValueError(f"start_x + num_squares_x must not exceed {_CODE_SIZE}.")
        if self.start_y + self.num_squares_y > _CODE_SIZE:  # Protect every horizontal and vertical code lookup.
            raise ValueError(f"start_y + num_squares_y must not exceed {_CODE_SIZE}.")

    def _make_point_data(self) -> np.ndarray:
        """Create flat pyCamSet object points indexed by global PuzzleBoard coordinates."""
        points = np.zeros((1, _CODE_SIZE * _CODE_SIZE, 3), dtype=np.float64)  # Match the one-board ChArUco shape.
        for row in range(_CODE_SIZE):  # Fill all possible detector row coordinates in the periodic field.
            for col in range(_CODE_SIZE):  # Fill all possible detector column coordinates in the periodic field.
                point_id = row * _CODE_SIZE + col  # Flatten the detector's (row, column) key for pyCamSet.
                points[0, point_id, 0] = (col - self.start_x) * self.square_size  # Use x across the printed board.
                points[0, point_id, 1] = (row - self.start_y) * self.square_size  # Use y down the printed board.
        return points  # Return a single face containing all globally addressable points.

    def _board_offsets(self) -> tuple[float, float]:
        """Return the centred SVG offset in millimetres."""
        off_x = (self.paper_width - self.num_squares_x * self.square_size) / 2.0  # Centre the board horizontally.
        off_y = (self.paper_height - self.num_squares_y * self.square_size) / 2.0  # Centre the board vertically.
        return off_x, off_y  # Return offsets in the SVG's millimetre coordinate system.

    def _svg_document(self) -> svgwrite.Drawing:
        """Construct the vector target using the exact loops from puzzle.js."""
        off_x, off_y = self._board_offsets()  # Compute the page-centred board origin.
        drawing = svgwrite.Drawing(  # Use physical millimetres in both SVG size and viewBox.
            size=(f"{self.paper_width:g}mm", f"{self.paper_height:g}mm"),
            viewBox=f"0 0 {self.paper_width:g} {self.paper_height:g}",
        )
        drawing.add(drawing.rect(insert=(0, 0), size=(self.paper_width, self.paper_height), fill="white"))  # Provide a white page.

        for x in range(self.num_squares_x + 1):  # Reproduce the generator's half-square outer boundary.
            for y in range(self.num_squares_y + 1):  # Reproduce the generator's half-square outer boundary.
                if (x + y + self.start_x + self.start_y) % 2 == 1:  # Skip the white checkerboard squares.
                    continue
                xx = x - 0.5  # Place the normal square relative to the corner grid.
                yy = y - 0.5  # Place the normal square relative to the corner grid.
                width = 1.0  # Start with a full square width.
                height = 1.0  # Start with a full square height.
                if x == 0:  # Trim the leftmost edge to a half square.
                    xx += 0.5
                    width = 0.5
                if y == 0:  # Trim the topmost edge to a half square.
                    yy += 0.5
                    height = 0.5
                if x == self.num_squares_x:  # Trim the rightmost edge to a half square.
                    width = 0.5
                if y == self.num_squares_y:  # Trim the bottommost edge to a half square.
                    height = 0.5
                drawing.add(drawing.rect(  # Add a black checkerboard square in millimetres.
                    insert=(off_x + xx * self.square_size, off_y + yy * self.square_size),
                    size=(width * self.square_size, height * self.square_size),
                    fill="black",
                    stroke="none",
                ))

        radius = self.square_size / 6.0  # Match RES_PER_SQ / 6 from the JavaScript generator.
        for y in range(self.num_squares_y):  # Generate horizontal-edge bits.
            for x in range(self.num_squares_x - 1):  # There is one bit between each adjacent pair of corners.
                value = int(_CODE_FIELD[self.start_y + y, self.start_x + x])  # Read the combined code value.
                fill = "white" if value & 2 else "black"  # Bit 2 encodes the horizontal edge.
                drawing.add(drawing.circle(  # Add a native SVG circle instead of a long PDF Bézier path.
                    center=(off_x + (1 + x) * self.square_size, off_y + (0.5 + y) * self.square_size),
                    r=radius,
                    fill=fill,
                    stroke="none",
                ))

        for y in range(self.num_squares_y - 1):  # Generate vertical-edge bits.
            for x in range(self.num_squares_x):  # Generate one vertical bit at each corner column.
                value = int(_CODE_FIELD[self.start_y + y, self.start_x + x])  # Read the combined code value.
                fill = "white" if value & 1 else "black"  # Bit 1 encodes the vertical edge.
                drawing.add(drawing.circle(  # Add a native SVG circle with physical millimetre coordinates.
                    center=(off_x + (0.5 + x) * self.square_size, off_y + (1 + y) * self.square_size),
                    r=radius,
                    fill=fill,
                    stroke="none",
                ))
        return drawing  # Return the complete vector document for saving or conversion.

    def save_to_svg(
        self,
        f_out: Path | str | None = None,
        suppress_svg_log: bool = False,
    ) -> Path:
        """Save the PuzzleBoard as a physically sized vector SVG."""
        if f_out is None:  # Construct a descriptive default filename when none is supplied.
            f_out = Path(f"puzzleboard_{self.num_squares_x}x{self.num_squares_y}_{self.square_size:g}mm.svg")
        else:  # Normalise caller-provided output paths in the same way as ChArUco.
            f_out = Path(f_out)
        f_out = f_out.expanduser().with_suffix(".svg").resolve()  # Force the correct vector extension.
        f_out.parent.mkdir(parents=True, exist_ok=True)  # Create the output directory when needed.
        svg_text = self._svg_document().tostring()  # Serialise the vector geometry once.
        with open(f_out, "w", encoding="utf-8", newline="\n") as fh:  # Write a normal saveable SVG file.
            fh.write(svg_text)  # Persist the complete SVG document.
            fh.flush()  # Flush Python's buffered file data.
        if (not f_out.exists()) or f_out.stat().st_size == 0:  # Verify that the requested file was created.
            raise IOError(f"SVG write failed: {f_out}")
        if not suppress_svg_log:  # Match ChArUco's optional logging behaviour.
            logging.info("Saved PuzzleBoard SVG: %s", f_out)
        return f_out  # Return the absolute output path.

    def save_to_pdf(
        self,
        f_out: Path | str | None = None,
        data_format: str = "raster",
        dpi: int = 300,
    ) -> Path:
        """Save a raster or vector PDF using the same API as the ChArUco target."""
        if f_out is None:  # Construct a descriptive default filename when none is supplied.
            f_out = Path(f"puzzleboard_{self.num_squares_x}x{self.num_squares_y}_{self.square_size:g}mm.pdf")
        else:  # Normalise caller-provided output paths before conversion.
            f_out = Path(f_out)
        f_out = f_out.expanduser().with_suffix(".pdf").resolve()  # Force the correct PDF extension.
        f_out.parent.mkdir(parents=True, exist_ok=True)  # Create the output directory when needed.
        svg_text = self._svg_document().tostring()  # Use one vector source for both PDF formats.
        if data_format == "vector":  # Preserve the original vector geometry in the PDF.
            cairosvg.svg2pdf(bytestring=svg_text.encode("utf-8"), write_to=str(f_out))  # Convert SVG paths without rasterising.
            logging.info("Saved PuzzleBoard Vector PDF: %s", f_out)  # Report the completed vector export.
            return f_out  # Return the generated PDF path.
        if data_format != "raster":  # Reject accidental format spellings rather than silently changing output.
            raise ValueError("data_format must be one of: raster, vector")
        png = cairosvg.svg2png(  # Render at the requested physical resolution for compatibility with ChArUco.
            bytestring=svg_text.encode("utf-8"),
            output_width=max(1, int(round(self.paper_width / 25.4 * dpi))),
            output_height=max(1, int(round(self.paper_height / 25.4 * dpi))),
        )
        with Image.open(BytesIO(png)) as image:  # Open the in-memory raster without a temporary file.
            image.convert("RGB").save(f_out, resolution=float(dpi))  # Save a conventional raster PDF.
        logging.info("Saved PuzzleBoard Raster PDF: %s", f_out)  # Report the completed raster export.
        return f_out  # Return the generated PDF path.

    def find_in_image(
        self,
        image,
        draw: bool = False,
        camera: Camera | None = None,
        wait_len: int = 1,
    ) -> ImageDetection:
        """Detect PuzzleBoard grid points and return pyCamSet-compatible IDs."""
        del camera  # PuzzleBoard's detector does not currently use camera intrinsics.
        point_ids, point_coords = detect_puzzleboard_image(image, min_width=self.min_width)  # Run the external detector.
        if not point_ids:  # Return the standard empty result when no board was found.
            return ImageDetection()
        positions = np.asarray(point_ids, dtype=np.int64)  # Convert decoded [row, column] positions to an array.
        coordinates = np.asarray(point_coords, dtype=np.float64)  # Convert detector image coordinates to an array.
        if positions.ndim != 2 or positions.shape[1] != 2:  # Guard against an incompatible detector version.
            raise ValueError("PuzzleBoard detector positions must have shape (n, 2).")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:  # Guard against malformed image coordinates.
            raise ValueError("PuzzleBoard detector coordinates must have shape (n, 2).")
        valid = (  # Keep only keys addressable by the finite point-data lookup.
            (positions[:, 0] >= 0)
            & (positions[:, 0] < _CODE_SIZE)
            & (positions[:, 1] >= 0)
            & (positions[:, 1] < _CODE_SIZE)
        )
        positions = positions[valid]  # Apply the key validity mask.
        coordinates = coordinates[valid]  # Apply the same mask to the image points.
        keys = positions[:, 0] * _CODE_SIZE + positions[:, 1]  # Flatten [row, column] into the point-data index.
        image_points = coordinates[:, ::-1]  # Detector coordinates are [row, column]; pyCamSet expects [x, y].
        if draw:  # Match ChArUco's optional visual debugging behaviour.
            display_im = np.asarray(image).copy()  # Avoid changing the caller's image.
            for x, y in image_points:  # Draw each accepted point in OpenCV's [x, y] order.
                cv2.circle(display_im, (int(round(x)), int(round(y))), 3, (0, 0, 255), 1)  # Mark the detection.
            cv2.imshow("PuzzleBoard detections", display_im)  # Display the annotated image.
            cv2.waitKey(wait_len)  # Honour the same wait-time convention as ChArUco.
        return ImageDetection(keys.astype(np.int64), image_points)  # Return IDs and image points to pyCamSet.

    def plot(self, imres: tuple[int, int] = (1000, 1000)) -> None:
        """Display a rasterised preview of the vector target."""
        png = cairosvg.svg2png(  # Rasterise only for interactive display; the saved target remains vector.
            bytestring=self._svg_document().tostring().encode("utf-8"),
            output_width=int(imres[0]),
            output_height=int(imres[1]),
        )
        with Image.open(BytesIO(png)) as image:  # Decode the in-memory preview.
            plt.imshow(np.asarray(image), cmap="gray")  # Display the target with matplotlib.
        plt.axis("off")  # Remove plot axes from the target preview.
        plt.show()  # Match the existing ChArUco plot method's interactive behaviour.


if __name__ == "__main__":
    test = PuzzleBoard()  # Preview the A4 2 mm default target when run directly.
    test.save_to_svg()  # Save a normal SVG beside the current working directory.
    # test.plot()  # Uncomment for an interactive preview.





