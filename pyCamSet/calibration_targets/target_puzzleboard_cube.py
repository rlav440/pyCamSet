'''
Purpose: Define a deterministic six-face PuzzleBoard cube target for pyCamSet.
Status: Active target with bounded face windows, cube geometry, and vector exports.
Future: Add alternative fixed layout versions only when backwards compatibility is preserved.
'''
from __future__ import annotations  # Keep postponed annotations consistent with pyCamSet targets.

from io import BytesIO  # Hold temporary SVG rasterisations in memory for PDF and visualisation output.
import logging  # Report completed cube exports.
from pathlib import Path  # Provide the same path handling as the other target generators.

import cairosvg  # Convert the vector net to PDF or raster preview data.
import cv2  # Draw optional detector results.
import numpy as np  # Store face origins, object points, and detected coordinates.
from PIL import Image  # Convert rasterised SVG data to PDF and texture arrays.
import svgwrite  # Write compact vector rectangles, polygons, and circles.

from pyCamSet.calibration_targets import AbstractTarget, FaceToShape, ImageDetection  # Reuse pyCamSet target contracts.
from pyCamSet.calibration_targets.puzzleboard_detection import detect_puzzleboard_image  # Use the credited PuzzleBoard detector.
from pyCamSet.calibration_targets.target_puzzleboard import _CODE_FIELD, _CODE_SIZE  # Reuse the exact generator code field.
from pyCamSet.cameras import Camera  # Keep the standard find_in_image signature.
from pyCamSet.utils.general_utils import make_4x4h_tform  # Convert cube face pose vectors to homogeneous transforms.


# These transforms use the same face order as the existing Ccube target: front, right, back, left, top, bottom.
TFORMS = [
    (([2.22144147, 2.22144147, 0.0]), ([-0.5, -0.5, 0.5])),
    (([-1.57079633, 0.0, 0.0]), ([-0.5, -0.5, 0.5])),
    (([-1.20919958, -1.20919958, 1.20919958]), ([0.5, -0.5, 0.5])),
    (([0.0, 2.22144147, -2.22144147]), ([0.5, 0.5, 0.5])),
    (([0.0, 0.0, 1.57079633]), ([0.5, -0.5, -0.5])),
    (([1.20919958, 1.20919958, 1.20919958]), ([-0.5, -0.5, -0.5])),
]

# These transforms unfold the six faces into a deterministic printable cube net.
NET_FORMS = [
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, -1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
]


CODE_LAYOUT_VERSION = "puzzle-cube-v1"  # Freeze the face-origin algorithm for reproducible cubes.
FACE_COUNT = 6  # A cube always has six faces.
FACE_GRID_COLUMNS = 3  # Pack front/right/back across the first code-field row.
FACE_GRID_ROWS = 2  # Pack left/top/bottom across the second code-field row.
FACE_WINDOW_GAP = 0  # Tile adjacent windows directly; their half-open ranges do not overlap.
MAX_FACE_SQUARES = (  # Enforce the six-face horizontal packing limit from the 501-position code field.
    _CODE_SIZE - (FACE_GRID_COLUMNS - 1) * FACE_WINDOW_GAP
) // FACE_GRID_COLUMNS


class PuzzleBoardCube(AbstractTarget):
    """Define a deterministic six-face PuzzleBoard calibration target."""

    def __init__(
        self,
        num_squares_per_side: int = 20,
        square_size: float = 10.0,
        min_width: int = 4,
        detection_options: dict | None = None,
    ):
        """Initialise a cube whose six faces use disjoint windows of the periodic code."""
        super().__init__(inputs=locals())  # Save constructor inputs for pyCamSet serialisation and multiprocessing.
        self.num_squares_per_side = int(num_squares_per_side)  # Store the square face dimension.
        self.square_size = float(square_size)  # Store the physical edge length in millimetres.
        self.min_width = int(min_width)  # Store the detector's minimum accepted grid width.
        self.detection_options = detection_options or {}  # Retain a future detector-options extension point.
        self.layout_version = CODE_LAYOUT_VERSION  # Record the deterministic face-layout version on the target.
        self.face_origins = self.face_origins_for_size(self.num_squares_per_side)  # Assign six disjoint code windows.
        self._validate_dimensions()  # Reject unsupported face sizes before constructing geometry.
        self.face_length = self.num_squares_per_side * self.square_size / 1000.0  # Convert cube side length to metres.
        self.base_face = np.array(  # Define the square face outline in the same order as Ccube.
            [[0.0, self.face_length, 0.0], [self.face_length, self.face_length, 0.0],
             [self.face_length, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        self.faceData = FaceToShape(  # Transform each local face into the common cube coordinate system.
            face_local_coords=self._make_local_face_points(),
            face_transforms=[make_4x4h_tform(*tform) for tform in TFORMS],
            scale_factor=self.face_length,
        )
        self.point_data = self.faceData.point_data  # Expose the transformed six-face geometry to pyCamSet.
        self._process_data()  # Compute pyCamSet's per-face local representation.

    @classmethod
    def face_origins_for_size(cls, num_squares_per_side: int) -> tuple[tuple[int, int], ...]:
        """Return deterministic row/column origins for the six disjoint face windows."""
        size = int(num_squares_per_side)  # Normalise the requested face size before arithmetic.
        if size < 2:  # A face needs at least two points to contain encoded edges.
            raise ValueError("num_squares_per_side must be at least 2.")
        if size > MAX_FACE_SQUARES:  # Six faces cannot fit disjointly beyond the bounded horizontal layout.
            raise ValueError(
                f"num_squares_per_side must not exceed {MAX_FACE_SQUARES} for {CODE_LAYOUT_VERSION}; "
                f"three face windows across must fit inside the {_CODE_SIZE}x{_CODE_SIZE} code field."
            )
        step = size + FACE_WINDOW_GAP  # Place adjacent windows consecutively without overlapping coordinates.
        origins = tuple(  # Use a stable row-major order matching front/right/back/left/top/bottom.
            (column * step, row * step)
            for row in range(FACE_GRID_ROWS)
            for column in range(FACE_GRID_COLUMNS)
        )
        if any(x + size > _CODE_SIZE or y + size > _CODE_SIZE for x, y in origins):  # Verify every window bound.
            raise ValueError("The deterministic face layout exceeds the PuzzleBoard code field.")
        return origins  # Return immutable origins so the layout cannot change accidentally.

    def _validate_dimensions(self) -> None:
        """Validate physical and detector parameters."""
        if self.num_squares_per_side < 2 or self.num_squares_per_side > MAX_FACE_SQUARES:  # Enforce the code limit.
            raise ValueError(f"num_squares_per_side must be between 2 and {MAX_FACE_SQUARES}.")
        if self.square_size <= 0.0:  # Physical geometry cannot use a zero or negative edge length.
            raise ValueError("square_size must be greater than zero.")
        if self.min_width < 1:  # The external detector requires a positive minimum width.
            raise ValueError("min_width must be at least 1.")

    def _make_local_face_points(self) -> np.ndarray:
        """Create the six identical local point grids before cube-face transforms."""
        size = self.num_squares_per_side  # Use one local square dimension for every face.
        side_m = self.face_length  # Use the physical face side in metres.
        points = np.zeros((FACE_COUNT, size * size, 3), dtype=np.float64)  # Allocate face/key-indexed points.
        for face in range(FACE_COUNT):  # Populate every face with the same local lattice.
            for row in range(size):  # Iterate local code rows.
                for column in range(size):  # Iterate local code columns.
                    point_id = row * size + column  # Flatten the local row/column key for pyCamSet.
                    points[face, point_id, 0] = column * side_m / size  # Place points at PuzzleBoard grid corners.
                    points[face, point_id, 1] = row * side_m / size  # Keep the physical face extent at side_m.
        return points  # Return local coordinates in metres.

    @staticmethod
    def _apply_affine(points_xy: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Apply a 3x3 affine matrix to an array of x/y points."""
        homogeneous = np.c_[points_xy, np.ones((len(points_xy), 1), dtype=float)]  # Add the affine homogeneous column.
        return (affine @ homogeneous.T).T[:, :2]  # Return transformed x/y points.

    def _net_affine_for_face(self, face_index: int) -> np.ndarray:
        """Convert one row/column net transform to an x/y metre transform."""
        face_form = np.asarray(NET_FORMS[face_index], dtype=float)  # Read the fixed face net transform.
        permute = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Convert row/column to x/y order.
        affine = permute @ face_form @ permute  # Match the existing Ccube net convention.
        affine[:2, 2] *= self.face_length  # Convert unit-face translations into metres.
        return affine  # Return the net transform for this face.

    def _face_rectangles(self, face_index: int) -> list[np.ndarray]:
        """Return black checkerboard polygons for one face in local metres."""
        size = self.num_squares_per_side  # Use the square face dimension.
        side_m = self.face_length  # Use the physical face side length.
        square_m = self.square_size / 1000.0  # Convert one printed square to metres.
        rectangles: list[np.ndarray] = []  # Collect only black squares, leaving the page background white.
        start_x, start_y = self.face_origins[face_index]  # Read this face's deterministic code origin.
        for x in range(size + 1):  # Preserve the JavaScript half-square boundary loop.
            for y in range(size + 1):  # Preserve the JavaScript half-square boundary loop.
                if (x + y + start_x + start_y) % 2 == 1:  # Skip white checkerboard cells.
                    continue
                xx = x - 0.5  # Start at the normal full-square position.
                yy = y - 0.5  # Start at the normal full-square position.
                width = 1.0  # Start with a full square width.
                height = 1.0  # Start with a full square height.
                if x == 0:  # Trim the left boundary to half a square.
                    xx += 0.5
                    width = 0.5
                if y == 0:  # Trim the top boundary to half a square.
                    yy += 0.5
                    height = 0.5
                if x == size:  # Trim the right boundary to half a square.
                    width = 0.5
                if y == size:  # Trim the bottom boundary to half a square.
                    height = 0.5
                rectangles.append(np.array(  # Store the black rectangle in local x/y metres.
                    [[xx * square_m, yy * square_m], [(xx + width) * square_m, yy * square_m],
                     [(xx + width) * square_m, (yy + height) * square_m], [xx * square_m, (yy + height) * square_m]],
                    dtype=float,
                ))
        return rectangles  # Return all black checkerboard polygons for this face.

    def _face_circles(self, face_index: int) -> list[tuple[np.ndarray, str]]:
        """Return encoded circle centres and colours for one face in local metres."""
        size = self.num_squares_per_side  # Use the square face dimension.
        square_m = self.square_size / 1000.0  # Convert one printed square to metres.
        radius_m = square_m / 6.0  # Match the JavaScript generator's RES_PER_SQ / 6 radius.
        start_x, start_y = self.face_origins[face_index]  # Read this face's deterministic code origin.
        circles: list[tuple[np.ndarray, str]] = []  # Store centre and fill for each encoded edge bit.
        for row in range(size):  # Generate horizontal-edge bit circles.
            for column in range(size - 1):  # There is one horizontal bit between adjacent columns.
                value = int(_CODE_FIELD[start_y + row, start_x + column])  # Read the combined code value.
                colour = "white" if value & 2 else "black"  # Bit 2 encodes the horizontal edge.
                circles.append((np.array([(column + 1) * square_m, (row + 0.5) * square_m]), colour))
        for row in range(size - 1):  # Generate vertical-edge bit circles.
            for column in range(size):  # There is one vertical bit at each column.
                value = int(_CODE_FIELD[start_y + row, start_x + column])  # Read the combined code value.
                colour = "white" if value & 1 else "black"  # Bit 1 encodes the vertical edge.
                circles.append((np.array([(column + 0.5) * square_m, (row + 1) * square_m]), colour))
        return circles  # Return all encoded circles for this face.

    def _add_face_geometry(
        self,
        drawing: svgwrite.Drawing,
        face_index: int,
        affine: np.ndarray,
        offset_m: np.ndarray,
    ) -> None:
        """Add one transformed face's checkerboard and code circles to an SVG drawing."""
        for rectangle in self._face_rectangles(face_index):  # Add each black checkerboard polygon.
            points_mm = self._apply_affine(rectangle, affine) * 1000.0 + offset_m * 1000.0  # Convert transformed metres to mm.
            drawing.add(drawing.polygon(points=[tuple(point) for point in points_mm], fill="black", stroke="none"))
        radius_mm = self.square_size / 6.0  # Convert the encoded circle radius directly to millimetres.
        for centre, colour in self._face_circles(face_index):  # Add each encoded circle.
            point_mm = self._apply_affine(centre.reshape(1, 2), affine)[0] * 1000.0 + offset_m * 1000.0
            drawing.add(drawing.circle(center=(float(point_mm[0]), float(point_mm[1])), r=radius_mm, fill=colour, stroke="none"))

    def _net_bounds(self, border_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the printable net bounds and page offset in metres."""
        outlines: list[np.ndarray] = []  # Collect transformed outer face corners.
        for face_index in range(FACE_COUNT):  # Evaluate all six fixed net positions.
            affine = self._net_affine_for_face(face_index)  # Read this face's net transform.
            outline = np.array([[0.0, 0.0], [self.face_length, 0.0], [self.face_length, self.face_length], [0.0, self.face_length]])
            outlines.append(self._apply_affine(outline, affine))  # Store transformed corners in metres.
        all_points = np.vstack(outlines)  # Combine all faces for one page extent.
        border_m = float(border_width) / 1000.0  # Convert requested border width from millimetres.
        min_xy = all_points.min(axis=0) - border_m  # Include the printable border around the net.
        max_xy = all_points.max(axis=0) + border_m  # Include the printable border around the net.
        offset_m = -min_xy  # Move the minimum net coordinate to the positive page border.
        return min_xy, max_xy, offset_m  # Return bounds and the translation used by SVG geometry.

    def _svg_document(
        self,
        border_width: float = 10.0,
        draw_cut_outline: bool = True,
        draw_face_ids: bool = False,
    ) -> tuple[svgwrite.Drawing, float, float]:
        """Construct a vector SVG of the six-face printable cube net."""
        min_xy, max_xy, offset_m = self._net_bounds(border_width)  # Calculate the page translation and dimensions.
        canvas_w_mm = float((max_xy[0] - min_xy[0]) * 1000.0)  # Convert net width to millimetres.
        canvas_h_mm = float((max_xy[1] - min_xy[1]) * 1000.0)  # Convert net height to millimetres.
        drawing = svgwrite.Drawing(  # Use millimetres as physical SVG units.
            size=(f"{canvas_w_mm:.6f}mm", f"{canvas_h_mm:.6f}mm"),
            viewBox=f"0 0 {canvas_w_mm:.6f} {canvas_h_mm:.6f}",
        )
        drawing.add(drawing.rect(insert=(0.0, 0.0), size=(canvas_w_mm, canvas_h_mm), fill="white"))  # Make white areas explicit.
        for face_index in range(FACE_COUNT):  # Draw all faces using their deterministic net transforms.
            affine = self._net_affine_for_face(face_index)  # Read this face's net transform.
            self._add_face_geometry(drawing, face_index, affine, offset_m)  # Add black squares and encoded circles.
            if draw_cut_outline:  # Add a thin outline to guide cutting and folding.
                outline = np.array([[0.0, 0.0], [self.face_length, 0.0], [self.face_length, self.face_length], [0.0, self.face_length]])
                points_mm = self._apply_affine(outline, affine) * 1000.0 + offset_m * 1000.0
                drawing.add(drawing.polygon(points=[tuple(point) for point in points_mm], fill="none", stroke="black", stroke_width=0.2))
            if draw_face_ids:  # Add optional assembly labels without changing the default calibration pattern.
                label = np.array([[0.02 * self.face_length, 0.985 * self.face_length]])  # Put the baseline in the bottom-left black border.
                label_mm = self._apply_affine(label, affine)[0] * 1000.0 + offset_m * 1000.0
                label_size_mm = self.face_length * 1000.0 * 0.045  # Keep printed label height proportional to the cube face.
                label_angle_deg = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))  # Follow the corresponding net-face orientation.
                label_text = drawing.text(  # Use viewBox millimetres directly; an additional mm suffix would rescale the text.
                    str(face_index + 1),
                    insert=tuple(label_mm),
                    fill="white",
                    font_size=f"{label_size_mm:.6f}",
                    font_family="Arial",
                    font_weight="bold",
                )
                label_text.rotate(label_angle_deg, center=tuple(label_mm))  # Keep each number upright relative to its face.
                drawing.add(label_text)  # Add the transformed label after the face geometry.
        return drawing, canvas_w_mm, canvas_h_mm  # Return the document and physical page dimensions.

    def _face_svg(self, face_index: int) -> str:
        """Create a square SVG texture for one face for 3-D visualisation."""
        side_mm = self.face_length * 1000.0  # Use the physical cube-face side in millimetres.
        drawing = svgwrite.Drawing(  # Create a face-local SVG texture.
            size=(f"{side_mm:.6f}mm", f"{side_mm:.6f}mm"),
            viewBox=f"0 0 {side_mm:.6f} {side_mm:.6f}",
        )
        drawing.add(drawing.rect(insert=(0.0, 0.0), size=(side_mm, side_mm), fill="white"))  # Fill white checkerboard areas.
        self._add_face_geometry(drawing, face_index, np.eye(3), np.zeros(2))  # Add the untransformed face pattern.
        drawing.add(drawing.text(  # Match Ccube by marking each visualised face with a human-readable number.
            str(face_index + 1),
            insert=(float(side_mm * 0.02), float(side_mm * 0.985)),  # Put the baseline in the bottom-left black border.
            fill="white",
            font_size=f"{side_mm * 0.045:.6f}",
            font_family="Arial",
            font_weight="bold",
        ))
        return drawing.tostring()  # Return SVG text for Cairo rasterisation.

    def save_to_svg(
        self,
        f_out: Path | str | None = None,
        border_width: float = 10.0,
        draw_cut_outline: bool = True,
        draw_face_ids: bool = True,
        suppress_svg_log: bool = False,
    ) -> Path:
        """Save the deterministic cube net as a physically sized vector SVG."""
        if f_out is None:  # Construct a descriptive filename when no output path was supplied.
            f_out = Path(f"puzzleboard_cube_{self.num_squares_per_side}_{self.square_size:g}mm.svg")
        else:  # Accept strings and paths like the other target classes.
            f_out = Path(f_out)
        f_out = f_out.expanduser().with_suffix(".svg").resolve()  # Force the vector extension.
        f_out.parent.mkdir(parents=True, exist_ok=True)  # Ensure the destination directory exists.
        drawing, _, _ = self._svg_document(border_width, draw_cut_outline, draw_face_ids)  # Build the vector document.
        svg_text = drawing.tostring()  # Serialise through svgwrite.
        with open(f_out, "w", encoding="utf-8", newline="\n") as fh:  # Persist a normal saveable SVG.
            fh.write(svg_text)  # Write all vector geometry.
            fh.flush()  # Flush buffered data before validation.
        if (not f_out.exists()) or f_out.stat().st_size == 0:  # Verify that output was created.
            raise IOError(f"SVG write failed: {f_out}")
        if not suppress_svg_log:  # Match the existing target exporters' logging behaviour.
            logging.info("Saved PuzzleBoard cube SVG: %s", f_out)
        return f_out  # Return the absolute saved path.

    def save_to_pdf(
        self,
        f_out: Path | str | None = None,
        data_format: str = "raster",
        dpi: int = 300,
        border_width: float = 10.0,
        draw_cut_outline: bool = True,
        draw_face_ids: bool = True,
    ) -> Path:
        """Save the cube net as a raster or vector PDF."""
        if f_out is None:  # Construct a descriptive filename when no output path was supplied.
            f_out = Path(f"puzzleboard_cube_{self.num_squares_per_side}_{self.square_size:g}mm.pdf")
        else:  # Accept strings and paths like the other target classes.
            f_out = Path(f_out)
        f_out = f_out.expanduser().with_suffix(".pdf").resolve()  # Force the PDF extension.
        f_out.parent.mkdir(parents=True, exist_ok=True)  # Ensure the destination directory exists.
        drawing, canvas_w_mm, canvas_h_mm = self._svg_document(border_width, draw_cut_outline, draw_face_ids)  # Build one SVG source.
        svg_bytes = drawing.tostring().encode("utf-8")  # Convert the svgwrite document to Cairo input.
        if data_format == "vector":  # Preserve all vector primitives in the resulting PDF.
            cairosvg.svg2pdf(bytestring=svg_bytes, write_to=str(f_out))  # Convert SVG without rasterising.
            logging.info("Saved PuzzleBoard cube Vector PDF: %s", f_out)  # Report the completed vector export.
            return f_out  # Return the generated PDF path.
        if data_format != "raster":  # Reject unsupported spellings explicitly.
            raise ValueError("data_format must be one of: raster, vector")
        png = cairosvg.svg2png(  # Rasterise at the requested physical resolution.
            bytestring=svg_bytes,
            output_width=max(1, int(round(canvas_w_mm / 25.4 * dpi))),
            output_height=max(1, int(round(canvas_h_mm / 25.4 * dpi))),
        )
        with Image.open(BytesIO(png)) as image:  # Decode the in-memory PNG.
            image.convert("RGB").save(f_out, resolution=float(dpi))  # Save a physical-size raster PDF.
        logging.info("Saved PuzzleBoard cube Raster PDF: %s", f_out)  # Report the completed raster export.
        return f_out  # Return the generated PDF path.

    def plot(self, return_scene: bool = False, draw_res: tuple[int, int] = (800, 800)):
        """Visualise the six textured faces in a 3-D pyVista scene."""
        textures: list[np.ndarray] = []  # Build one raster texture per face only when visualisation is requested.
        for face_index in range(FACE_COUNT):  # Rasterise each deterministic face pattern.
            png = cairosvg.svg2png(  # Use the requested preview resolution for each face texture.
                bytestring=self._face_svg(face_index).encode("utf-8"),
                output_width=int(draw_res[0]),
                output_height=int(draw_res[1]),
            )
            with Image.open(BytesIO(png)) as image:  # Decode the temporary texture.
                textures.append(np.asarray(image.convert("RGB")))  # Store an RGB texture for FaceToShape.
        scene = self.faceData.draw_meshes(self.base_face, textures, return_scene=return_scene)  # Use pyCamSet cube rendering.
        if return_scene:  # Let callers embed or further configure the pyVista scene.
            return scene
        return None  # draw_meshes displays the scene when return_scene is false.

    def find_in_image(
        self,
        image,
        draw: bool = False,
        camera: Camera | None = None,
        wait_len: int = 1,
    ) -> ImageDetection:
        """Detect PuzzleBoard points and assign each point to its deterministic cube face."""
        del camera  # The current PuzzleBoard detector does not use camera intrinsics.
        point_ids, point_coords = detect_puzzleboard_image(image, min_width=self.min_width)  # Decode global code positions.
        if len(point_ids) == 0:  # Return the standard empty result when no face was found.
            return ImageDetection()
        positions = np.asarray(point_ids, dtype=np.int64)  # Detector positions are [row, column].
        coordinates = np.asarray(point_coords, dtype=np.float64)  # Detector image coordinates are [row, column].
        if positions.ndim != 2 or positions.shape[1] != 2:  # Guard against incompatible detector versions.
            raise ValueError("PuzzleBoard detector positions must have shape (n, 2).")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:  # Guard against malformed image coordinates.
            raise ValueError("PuzzleBoard detector coordinates must have shape (n, 2).")
        keys: list[list[int]] = []  # Store [face_index, local_flat_point] keys for pyCamSet.
        image_points: list[np.ndarray] = []  # Store matching [x, y] image coordinates.
        size = self.num_squares_per_side  # Use the configured local face dimension.
        for position, coordinate in zip(positions, coordinates):  # Classify every recovered global point.
            row, column = int(position[0]), int(position[1])  # Preserve the detector's row/column convention.
            matches = []  # Collect any face windows containing this global coordinate.
            for face_index, (origin_x, origin_y) in enumerate(self.face_origins):  # Search the six fixed windows.
                local_column = column - origin_x  # Convert global code column to a face-local column.
                local_row = row - origin_y  # Convert global code row to a face-local row.
                if 0 <= local_column < size and 0 <= local_row < size:  # Accept only points inside one face window.
                    matches.append((face_index, local_row, local_column))
            if len(matches) != 1:  # Reject guard-band or ambiguous points rather than assigning the wrong face.
                continue
            face_index, local_row, local_column = matches[0]  # Unpack the unique face classification.
            keys.append([face_index, local_row * size + local_column])  # Flatten the local key within that face.
            image_points.append(coordinate[::-1])  # Convert detector [row, column] to pyCamSet [x, y].
        if not keys:  # Return no data if the decoded points belong only to guard bands.
            return ImageDetection()
        image_points_array = np.asarray(image_points, dtype=np.float64)  # Convert accepted image points to an array.
        if draw:  # Match the visual debugging behaviour of the other targets.
            display_im = np.asarray(image).copy()  # Avoid changing the caller's image.
            for point in image_points_array:  # Draw accepted points in OpenCV [x, y] order.
                cv2.circle(display_im, (int(round(point[0])), int(round(point[1]))), 3, (0, 0, 255), 1)
            cv2.imshow("PuzzleBoard cube detections", display_im)  # Show classified detections.
            cv2.waitKey(wait_len)  # Honour the standard wait-time convention.
        return ImageDetection(np.asarray(keys, dtype=np.int64), image_points_array)  # Return face-aware detections.


if __name__ == "__main__":
    test = PuzzleBoardCube()  # Construct the deterministic default cube.
    test.save_to_svg()  # Save a printable SVG net in the current directory.
    # test.plot()  # Uncomment for an interactive 3-D preview.









