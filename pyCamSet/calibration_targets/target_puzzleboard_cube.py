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

# True cube opposite-face pairs, derived from the TFORMS geometry by computing
# each face's outward world normal (R @ [0,0,1]) and pairing anti-parallel
# faces (dot == -1). This is NOT the formula (face + 3) % 6, which wrongly gives
# (0,3),(1,4),(2,5) — only (2,5) is correct under that formula. The true pairs
# are (0,4),(1,3),(2,5). Verified by hermes_relgeom_round3_opposite_derivation.py
# (all pairs anti-parallel, symmetric, dot == -1.0000). Frozen as a constant so
# production, validation scripts, and tests share one source of truth.
OPPOSITE_FACE_MAP = {0: 4, 1: 3, 2: 5, 3: 1, 4: 0, 5: 2}

# Minimum effective full-field FOV (degrees) under which face reassignment must
# refuse to fire. Round 3 wide-FOV sweep (hermes_relgeom_round3_moderate_issues.py,
# 60 poses/bin, n_sq=6, conf>3.0 on a 500px-wide image) showed fire-accuracy
# drops BELOW the 0.25 chance baseline to 0.28-0.33 once the effective full FOV
# falls below ~13deg (fx > ~2000 on a 500px image): bins 2275-2517 (FOV 11.9deg,
# fire_acc 0.333) and 2517-2758 (FOV 10.8deg, fire_acc 0.281) are inverted. The
# safe regime is fx ~340-1550 (FOV ~20-57deg), where fire_acc stays above 0.76.
# The guard computes the effective full FOV from the assumed fx and the image
# width at detection time and, below this threshold, drops the contaminated
# cluster via the gate's existing fail-safe path instead of running the PnP
# tie-breaker — never force an assignment the signal is known to get wrong more
# often than not. See REBELS_RELATIVE_GEOMETRY_FACE_ID_REPORT.md "Moderate
# issues" §1 and "Round 4 — final cleanup" §Issue 3.
FACE_REASSIGNMENT_MIN_SAFE_FOV_DEG = 13.0

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
        plane_consistency_gate: bool = False,
        plane_gate_inlier_squares: float = 0.5,
        plane_gate_contam_squares: float = 2.0,
        plane_gate_min_contam_frac: float = 0.25,
        plane_gate_min_points: int = 8,
        plane_gate_ransac_iters: int = 200,
        plane_gate_random_state: int = 0,
        face_reassignment: bool = False,
        face_reassignment_confidence: float = 3.0,
        face_reassignment_intrinsics_fx: float | None = None,
        face_reassignment_intrinsics_cx: float | None = None,
        face_reassignment_intrinsics_cy: float | None = None,
        detection_options: dict | None = None,
    ):
        """Initialise a cube whose six faces use disjoint windows of the periodic code.

        plane_consistency_gate (default False, opt-in): when True, ``find_in_image``
        runs a two-stage RANSAC homography consistency check per face and drops
        points that are not geometrically consistent with the face's majority plane.
        Designed to catch geometrically impossible merged point sets (two physical
        regions of the cube incorrectly decoded into the same face window). Default
        OFF until validated against the full component survey; see
        ``REBELS_PLANE_CONSISTENCY_GATE_IMPLEMENTATION_REPORT.md``.

        face_reassignment (default False, opt-in): when True (and
        ``plane_consistency_gate`` is also True), the dropped cluster from each
        contaminated face is not discarded outright — instead a joint two-cluster
        PnP identifies which of the cube's other co-visible faces the dropped
        cluster most likely belongs to, and if the PnP confidence exceeds
        ``face_reassignment_confidence`` (second-best/best reprojection-error
        ratio), the dropped points are relabelled to that face and kept. If the
        signal is not confident, the points are dropped (the gate's existing
        behaviour) — never force an assignment the signal does not support. This
        is the validated method (b) from the round-2 synthetic validation
        (``hermes_relgeom_face_id_round2_validation.py``): raw 60-80% accuracy
        across noise 0-2px, grids 4-8, confident(>3.0) 93.3% on ~48% of trials,
        confident(>8.0) 98.4% on ~43%. The intrinsics used are a generic/assumed
        estimate (per the project's standing constraint that calibration output
        must never be a functional input to detection): by default a focal length
        derived from a typical sensor-size/FOV assumption (fx = image_width *
        1.4, a moderate wide-ish lens), principal point at the image centre.
        Override via the ``face_reassignment_intrinsics_*`` parameters if a
        different generic estimate is preferred. NEVER load these from a
        ``.camset`` file or other calibration output.

        Narrow-FOV safety: the PnP confidence gate is known to invert at
        effective full-field FOV below
        ``FACE_REASSIGNMENT_MIN_SAFE_FOV_DEG`` (~13deg; fx > ~2000 on a 500px
        image) — fire-accuracy drops below the 0.25 chance baseline there, so
        enabling ``face_reassignment=True`` with a telephoto/narrow-FOV setup
        would produce confidently-wrong relabelings more often than not. The
        method therefore refuses to run the PnP tie-breaker in that regime and
        falls back to the gate's drop path (the dropped cluster is dropped,
        never force-assigned). See ``_run_face_reassignment`` and
        ``REBELS_RELATIVE_GEOMETRY_FACE_ID_REPORT.md`` "Round 4 — final cleanup"
        §Issue 3.

        Gate thresholds (in units of square pitch):
        - plane_gate_contam_squares (2.0): stage-1 trigger — a face is considered
          contaminated only if >= plane_gate_min_contam_frac of its points have
          residual > this value under the all-points homography. 2.0 squares sits
          above the max residual of clean single planes (0.69-1.50 squares) and
          below the median residual of contaminated merges (1.3-2.2 squares).
        - plane_gate_min_contam_frac (0.25): minimum fraction of high-residual
          points required to trigger stage 2. Clean images have 0% above 2.0
          squares; contaminated images have 25-64%.
        - plane_gate_inlier_squares (0.5): stage-2 RANSAC inlier threshold. Tight
          enough to reject foreign-plane points that a loose 1.0-square threshold
          admits (the round-2 bug: on 915/91503 a 1.0-square threshold let 6
          foreign points fit the wrong cluster at 0.45-1.07 squares, so max-count
          RANSAC picked a 30-point wrong model over the correct 24-point model).
          The true plane's inliers fit at 0.02-0.45 squares after RANSAC; 0.5
          squares separates the two planes cleanly. Stage-1 still uses
          plane_gate_contam_squares (2.0) so clean single planes (max residual
          0.69-1.50 squares) never reach stage 2.
        """
        super().__init__(inputs=locals())  # Save constructor inputs for pyCamSet serialisation and multiprocessing.
        self.num_squares_per_side = int(num_squares_per_side)  # Store the square face dimension.
        self.square_size = float(square_size)  # Store the physical edge length in millimetres.
        self.min_width = int(min_width)  # Store the detector's minimum accepted grid width.
        self.plane_consistency_gate = bool(plane_consistency_gate)  # Opt-in geometric gate.
        self.plane_gate_inlier_squares = float(plane_gate_inlier_squares)  # RANSAC inlier threshold (pitch units).
        self.plane_gate_contam_squares = float(plane_gate_contam_squares)  # Stage-1 contamination trigger (pitch units).
        self.plane_gate_min_contam_frac = float(plane_gate_min_contam_frac)  # Min fraction of high-residual points to trigger.
        self.plane_gate_min_points = int(plane_gate_min_points)  # Minimum face population to run the gate.
        self.plane_gate_ransac_iters = int(plane_gate_ransac_iters)  # RANSAC iterations.
        self.plane_gate_random_state = int(plane_gate_random_state)  # Deterministic seed.
        self.face_reassignment = bool(face_reassignment)  # Opt-in face reassignment (requires plane_consistency_gate).
        self.face_reassignment_confidence = float(face_reassignment_confidence)  # PnP confidence threshold (second/best ratio).
        # Generic/assumed intrinsics overrides (None -> derive from image size at detection time).
        # Per the project standing constraint, these are NEVER loaded from calibration output.
        self.face_reassignment_intrinsics_fx = (float(face_reassignment_intrinsics_fx)
                                                if face_reassignment_intrinsics_fx is not None else None)
        self.face_reassignment_intrinsics_cx = (float(face_reassignment_intrinsics_cx)
                                                 if face_reassignment_intrinsics_cx is not None else None)
        self.face_reassignment_intrinsics_cy = (float(face_reassignment_intrinsics_cy)
                                                 if face_reassignment_intrinsics_cy is not None else None)
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

    def _run_plane_consistency_gate(
        self,
        keys: list[list[int]],
        image_points: list[np.ndarray],
    ) -> tuple[list[list[int]], list[np.ndarray]]:
        """Drop face points that are inconsistent with their face's majority plane.

        Two-stage gate per face:

        Stage 1 (contamination trigger): fit a single homography to ALL of the
        face's (local_id -> pixel) points. If fewer than ``plane_gate_min_contam_frac``
        of the points have residual > ``plane_gate_contam_squares * pitch``, the face
        is a clean single plane (possibly with lens distortion) → return unchanged.
        This prevents false positives on lens-distorted single planes whose edge
        points deviate by 0.5-1.5 squares but are still one physical plane.

        Stage 2 (RANSAC split): if the trigger fires, run RANSAC — repeatedly sample
        a minimal 4-point subset, fit a homography, count inliers (residual <
        ``plane_gate_inlier_squares * pitch``), keep the best-inlier model, refit on
        all inliers, and drop outliers. The refit re-evaluates every point under the
        refined homography with the same inlier threshold, which can recover a few
        extra inliers that the initial RANSAC missed due to a suboptimal minimal
        sample. No separate stability guard is applied: on the flagship contaminated
        image (915/91503) the pre-refit self-fit max (0.19 sq) is well below the
        0.5 sq inlier threshold, so a ``min(pre_refit_max, inlier_thresh)`` cap would
        be inert (no point lies in that band), and the plain threshold alone produces
        the correct 24-point cluster with 0 cross-plane leaks. The cross-plane leak
        fix is attributable entirely to the 0.5 sq threshold tightening.

        This corrects the two blocking flaws in Fable5's original design:
        (1) adjacency clustering cannot split two true planes whose components share
        a 4-adjacent boundary point (2 of the 3 contaminated images);
        (2) a per-component residual threshold is contradicted by the survey data
        (clean counterexamples at 0.8-1.13 squares; contaminated components at
        0.03-0.22 squares each). RANSAC on the *merged* face point set handles
        both, because it ignores id adjacency and measures only geometric
        consistency under a single homography. The two-stage trigger prevents
        false positives on lens-distorted single planes (max residual 0.69-1.50
        squares, 0% above 2.0 squares) while catching genuine contamination
        (25-64% of points above 2.0 squares). The 0.5-square stage-2 inlier
        threshold (tightened from 1.0 in round 2) prevents max-count RANSAC from
        selecting a larger-but-wrong consensus that includes foreign points
        fitting the wrong cluster at 0.45-1.07 squares.

        Deterministic: fixed ``plane_gate_random_state`` seed. Fails open — if a
        homography cannot be fit, the face is returned unchanged.
        """
        if not keys:
            return keys, image_points

        size = self.num_squares_per_side
        inlier_thresh_squares = self.plane_gate_inlier_squares
        contam_thresh_squares = self.plane_gate_contam_squares
        min_contam_frac = self.plane_gate_min_contam_frac
        min_points = self.plane_gate_min_points
        n_iters = self.plane_gate_ransac_iters

        # Group indices by face.
        face_to_idx: dict[int, list[int]] = {}
        for i, k in enumerate(keys):
            face_to_idx.setdefault(int(k[0]), []).append(i)

        keep_mask = [True] * len(keys)
        for face_index, idxs in face_to_idx.items():
            if len(idxs) < min_points:
                continue  # too few points to fit a homography meaningfully

            # Build (local_col, local_row) source points and (x, y) dest points.
            src = []
            dst = []
            for i in idxs:
                local_flat = int(keys[i][1])
                local_row, local_col = divmod(local_flat, size)
                src.append([float(local_col), float(local_row)])
                dst.append([float(image_points[i][0]), float(image_points[i][1])])
            src = np.asarray(src, dtype=np.float64)
            dst = np.asarray(dst, dtype=np.float64)
            n = len(idxs)
            if n < 4:
                continue

            # Estimate pitch = median distance among id-adjacent point pairs.
            pitch = self._estimate_face_pitch(src, dst)
            if pitch is None or pitch <= 0 or not np.isfinite(pitch):
                continue  # cannot establish a scale — fail open
            contam_thresh_px = contam_thresh_squares * pitch
            inlier_thresh_px = inlier_thresh_squares * pitch

            # Stage 1: fit homography to ALL face points, check contamination trigger.
            H_all, _ = cv2.findHomography(src, dst, method=0)
            if H_all is None:
                continue  # cannot fit — fail open
            dst_pred_all = cv2.perspectiveTransform(src.reshape(1, -1, 2), H_all)[0]
            errs_all = np.linalg.norm(dst - dst_pred_all, axis=1)
            contam_frac = float(np.mean(errs_all > contam_thresh_px))
            if contam_frac < min_contam_frac:
                continue  # clean single plane (possibly with lens distortion) — no filtering

            # Stage 2: RANSAC to find the majority plane and drop outliers.
            # Use OpenCV's built-in RANSAC (method=cv2.RANSAC) — battle-tested
            # with proper near-collinear/near-duplicate degeneracy checking.
            # NOTE: cv2.RANSAC's CheckSubset only rejects near-collinear samples
            # within a single 4-point draw; it has no notion of "which real-world
            # plane a point came from" and accepts mixed 3+1 cross-plane samples
            # (verified empirically: 300/300 accepted). The split correctness
            # therefore depends on the inlier threshold, not on cv2's degeneracy
            # checking. A loose 1.0-square threshold admitted 6 foreign points on
            # 915/91503 (they fit the wrong cluster at 0.45-1.07 squares), so
            # max-count RANSAC picked a 30-point wrong model over the correct
            # 24-point model; 0.5 squares separates the two planes cleanly.
            H_ransac, ransac_mask = cv2.findHomography(
                src, dst, method=cv2.RANSAC, confidence=0.999,
                maxIters=n_iters, ransacReprojThreshold=inlier_thresh_px,
            )
            if H_ransac is None or ransac_mask is None:
                continue  # no valid homography — fail open
            best_inlier_mask = ransac_mask.ravel().astype(bool)
            best_inlier_count = int(best_inlier_mask.sum())

            # Refit on all inliers to refine, then re-evaluate with the same
            # inlier threshold. This can recover a few extra inliers that the
            # initial RANSAC missed due to a slightly suboptimal minimal sample.
            # No separate stability guard: a ``min(pre_refit_max, inlier_thresh)``
            # cap is inert on 915/91503 (pre-refit max 0.19 sq < 0.5 sq thresh,
            # so no point sits in the band the cap would remove), and the plain
            # threshold alone gives the correct 24-point cluster with 0 leaks.
            # A ``max(pre_refit_max, inlier_thresh)`` cap (the round-2 code) is
            # always inert by construction: max(a, b) >= b, so for non-negative
            # residuals (x < b) & (x <= max(a, b)) == (x < b) — the second
            # clause excludes nothing the first didn't already exclude.
            if best_inlier_count >= 4:
                H_ref, _ = cv2.findHomography(
                    src[best_inlier_mask], dst[best_inlier_mask], method=0
                )
                if H_ref is not None:
                    dst_pred = cv2.perspectiveTransform(src.reshape(1, -1, 2), H_ref)[0]
                    errs = np.linalg.norm(dst - dst_pred, axis=1)
                    best_inlier_mask = errs < inlier_thresh_px

            # Guard: if the best model has fewer than min_points inliers, the face
            # is not a clean single plane — leave unfiltered (fail open).
            if int(np.sum(best_inlier_mask)) < min_points:
                continue

            for local_i, global_i in enumerate(idxs):
                if not best_inlier_mask[local_i]:
                    keep_mask[global_i] = False

        filtered_keys = [k for k, keep in zip(keys, keep_mask) if keep]
        filtered_image_points = [p for p, keep in zip(image_points, keep_mask) if keep]
        return filtered_keys, filtered_image_points

    def _run_face_reassignment(
        self,
        keys: list[list[int]],
        image_points: list[np.ndarray],
        image_shape: tuple[int, int] | None = None,
    ) -> tuple[list[list[int]], list[np.ndarray]]:
        """Reassign dropped-cluster points to their true cube face via joint PnP.

        This mirrors ``_run_plane_consistency_gate``'s two-stage RANSAC split
        (stage-1 contamination trigger, stage-2 RANSAC majority-plane selection
        with the same 0.5-square inlier threshold) but, instead of discarding
        the dropped cluster, attempts to identify which of the cube's other
        co-visible faces it belongs to using the validated method (b) from
        ``hermes_relgeom_face_id_round2_validation.py``: a single
        ``cv2.solvePnP`` per candidate face hypothesis using ALL corner points
        from both the kept cluster (at known local coords in the kept face's
        frame, transformed to world via TFORMS) and the hypothesized-dropped
        cluster (placed via TFORMS for that hypothesis), both matched against
        the same measured 2D points. The hypothesis with the lowest PnP
        reprojection error wins; confidence = second-best / best error ratio.
        If confidence >= ``face_reassignment_confidence``, the dropped points
        are relabelled to the winning face and kept; otherwise they are dropped
        (the gate's existing behaviour).

        The camera intrinsics K used here is a generic/assumed estimate, never
        loaded from calibration output (per the project's standing constraint).
        By default fx = image_width * 1.4 (a moderate wide-ish lens), principal
        point at the image centre; override via the
        ``face_reassignment_intrinsics_*`` constructor parameters.

        This method is additive to ``_run_plane_consistency_gate`` and does
        not modify its logic. It runs instead of the gate when
        ``face_reassignment=True``.

        Narrow-FOV safety guard: the PnP confidence gate is known to invert
        (fire-accuracy drops *below* the 0.25 chance baseline, i.e. it gets the
        relabeling wrong more often than not) at effective full-field FOV below
        ~13deg (fx > ~2000 on a 500px-wide image). See
        ``REBELS_RELATIVE_GEOMETRY_FACE_ID_REPORT.md`` "Moderate issues" §1 and
        "Round 4 — final cleanup" §Issue 3. When the assumed fx and the
        detection-time image width imply a full FOV below
        ``FACE_REASSIGNMENT_MIN_SAFE_FOV_DEG``, this method refuses to run the
        PnP tie-breaker entirely and instead falls back to the gate's existing
        drop path for every contaminated face (the dropped cluster is dropped,
        never force-assigned). This matches the "never force an assignment the
        signal does not support" design philosophy.
        """
        if not keys:
            return keys, image_points

        size = self.num_squares_per_side
        inlier_thresh_squares = self.plane_gate_inlier_squares
        contam_thresh_squares = self.plane_gate_contam_squares
        min_contam_frac = self.plane_gate_min_contam_frac
        min_points = self.plane_gate_min_points
        n_iters = self.plane_gate_ransac_iters
        conf_thresh = self.face_reassignment_confidence

        # Generic/assumed intrinsics (never from calibration output).
        if image_shape is not None and len(image_shape) >= 2:
            h_img, w_img = int(image_shape[0]), int(image_shape[1])
        else:
            # Fallback: estimate from the point cloud extent.
            if image_points:
                xs = np.asarray([p[0] for p in image_points])
                ys = np.asarray([p[1] for p in image_points])
                w_img = int(max(xs.max(), 1)) if len(xs) else 500
                h_img = int(max(ys.max(), 1)) if len(ys) else 400
            else:
                w_img, h_img = 500, 400
        fx = self.face_reassignment_intrinsics_fx if self.face_reassignment_intrinsics_fx is not None else float(w_img) * 1.4
        cx = self.face_reassignment_intrinsics_cx if self.face_reassignment_intrinsics_cx is not None else w_img / 2.0
        cy = self.face_reassignment_intrinsics_cy if self.face_reassignment_intrinsics_cy is not None else h_img / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        # Narrow-FOV safety guard (Round 4 Issue 3). The effective full-field
        # FOV is derivable from the assumed fx and the detection-time image
        # width as 2 * arctan((w/2) / fx). Below FACE_REASSIGNMENT_MIN_SAFE_FOV_DEG
        # the PnP confidence gate is known to invert (fire-accuracy < chance
        # baseline), so we must NOT run the tie-breaker — drop the dropped
        # cluster via the gate's existing fail-safe path instead. The guard
        # uses the *effective* FOV at detection time, not the raw fx value, so
        # it correctly classifies a telephoto setup regardless of the user's
        # assumed-fx override or image resolution. Safe-regime firing (fx
        # ~340-1550, FOV ~20-57deg) is unaffected: 13deg is well below that
        # band's lower edge.
        if fx > 0 and w_img > 0:
            full_fov_deg = 2.0 * float(np.degrees(np.arctan((w_img / 2.0) / fx)))
            if full_fov_deg < FACE_REASSIGNMENT_MIN_SAFE_FOV_DEG:
                # Refuse to fire the reassignment in the known-bad narrow-FOV
                # regime. Reuse the gate's two-stage split + drop path so the
                # dropped cluster is still removed (not forced) — only the PnP
                # tie-breaker is skipped. This is the same fail-safe the gate
                # uses when confidence is low or candidates are insufficient.
                return self._run_plane_consistency_gate(keys, image_points)

        # Precompute the six faces' world point lattices (TFORMS composition).
        # Local id ordering is row*size + col, matching _make_local_face_points.
        side_m = self.face_length
        tform_matrices = [make_4x4h_tform(*t) for t in TFORMS]
        local_lattice = np.zeros((size * size, 3), dtype=np.float64)
        for row in range(size):
            for col in range(size):
                local_lattice[row * size + col, 0] = col * side_m / size
                local_lattice[row * size + col, 1] = row * side_m / size
        face_world_lattices = []
        for f in range(6):
            M = tform_matrices[f]
            R = M[:3, :3]
            t = M[:3, 3]
            face_world_lattices.append((R @ local_lattice.T).T + t)

        # Group indices by face.
        face_to_idx: dict[int, list[int]] = {}
        for i, k in enumerate(keys):
            face_to_idx.setdefault(int(k[0]), []).append(i)

        keep_mask = [True] * len(keys)
        reassigned_keys: list[list[int]] = list(keys)  # Copy for relabelling.
        for face_index, idxs in face_to_idx.items():
            if len(idxs) < min_points:
                continue

            src = []
            dst = []
            for i in idxs:
                local_flat = int(keys[i][1])
                local_row, local_col = divmod(local_flat, size)
                src.append([float(local_col), float(local_row)])
                dst.append([float(image_points[i][0]), float(image_points[i][1])])
            src = np.asarray(src, dtype=np.float64)
            dst = np.asarray(dst, dtype=np.float64)
            n = len(idxs)
            if n < 4:
                continue

            pitch = self._estimate_face_pitch(src, dst)
            if pitch is None or pitch <= 0 or not np.isfinite(pitch):
                continue
            contam_thresh_px = contam_thresh_squares * pitch
            inlier_thresh_px = inlier_thresh_squares * pitch

            # Stage 1: contamination trigger.
            H_all, _ = cv2.findHomography(src, dst, method=0)
            if H_all is None:
                continue
            dst_pred_all = cv2.perspectiveTransform(src.reshape(1, -1, 2), H_all)[0]
            errs_all = np.linalg.norm(dst - dst_pred_all, axis=1)
            contam_frac = float(np.mean(errs_all > contam_thresh_px))
            if contam_frac < min_contam_frac:
                continue  # clean single plane — no reassignment needed.

            # Stage 2: RANSAC split (same as the gate).
            H_ransac, ransac_mask = cv2.findHomography(
                src, dst, method=cv2.RANSAC, confidence=0.999,
                maxIters=n_iters, ransacReprojThreshold=inlier_thresh_px,
            )
            if H_ransac is None or ransac_mask is None:
                continue
            best_inlier_mask = ransac_mask.ravel().astype(bool)
            best_inlier_count = int(best_inlier_mask.sum())
            if best_inlier_count >= 4:
                H_ref, _ = cv2.findHomography(
                    src[best_inlier_mask], dst[best_inlier_mask], method=0
                )
                if H_ref is not None:
                    dst_pred = cv2.perspectiveTransform(src.reshape(1, -1, 2), H_ref)[0]
                    errs = np.linalg.norm(dst - dst_pred, axis=1)
                    best_inlier_mask = errs < inlier_thresh_px
            if int(np.sum(best_inlier_mask)) < min_points:
                continue

            kept_local_ids = [int(keys[idxs[i]][1]) for i in range(n) if best_inlier_mask[i]]
            dropped_local_ids = [int(keys[idxs[i]][1]) for i in range(n) if not best_inlier_mask[i]]
            if len(kept_local_ids) < 4 or len(dropped_local_ids) < 4:
                # Too few points in either cluster for a meaningful PnP.
                for local_i, global_i in enumerate(idxs):
                    if not best_inlier_mask[local_i]:
                        keep_mask[global_i] = False
                continue

            # Method (b): joint two-cluster PnP per candidate face hypothesis.
            kept_img = np.asarray([dst[i] for i in range(n) if best_inlier_mask[i]], dtype=np.float64)
            dropped_img = np.asarray([dst[i] for i in range(n) if not best_inlier_mask[i]], dtype=np.float64)
            img2d = np.vstack([kept_img, dropped_img]).astype(np.float64)

            # Candidate dropped faces: all faces except the kept face and its
            # true geometrically-derived opposite (never co-visible). The
            # opposite is taken from OPPOSITE_FACE_MAP, which is derived from
            # the TFORMS geometry (anti-parallel outward normals), NOT the
            # (face + 3) % 6 formula that is wrong for 4 of 6 faces. Round 3
            # issue 1 fix.
            opposite = OPPOSITE_FACE_MAP[face_index]
            candidates = [f for f in range(6) if f != face_index and f != opposite]
            kept_obj = face_world_lattices[face_index][kept_local_ids]
            candidate_errors = []
            for X in candidates:
                dropped_obj = face_world_lattices[X][dropped_local_ids]
                obj3d = np.vstack([kept_obj, dropped_obj]).astype(np.float64)
                if len(obj3d) < 4:
                    continue
                ok, rvec, tvec = cv2.solvePnP(
                    obj3d, img2d, K, None, flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok:
                    continue
                proj, _ = cv2.projectPoints(obj3d, rvec, tvec, K, None)
                err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img2d, axis=1)))
                candidate_errors.append((err, X))
            if len(candidate_errors) < 2:
                # Not enough candidates to form a confidence ratio — drop.
                for local_i, global_i in enumerate(idxs):
                    if not best_inlier_mask[local_i]:
                        keep_mask[global_i] = False
                continue
            candidate_errors.sort(key=lambda x: x[0])
            best_err, best_X = candidate_errors[0]
            second_err = candidate_errors[1][0]
            confidence = second_err / max(best_err, 1e-12)
            if confidence >= conf_thresh:
                # Confident: relabel the dropped cluster to the winning face.
                dropped_global_idxs = [idxs[i] for i in range(n) if not best_inlier_mask[i]]
                for gi in dropped_global_idxs:
                    reassigned_keys[gi] = [best_X, reassigned_keys[gi][1]]
            else:
                # Not confident: drop (gate's existing behaviour).
                for local_i, global_i in enumerate(idxs):
                    if not best_inlier_mask[local_i]:
                        keep_mask[global_i] = False

        filtered_keys = [k for k, keep in zip(reassigned_keys, keep_mask) if keep]
        filtered_image_points = [p for p, keep in zip(image_points, keep_mask) if keep]
        return filtered_keys, filtered_image_points

    @staticmethod
    def _estimate_face_pitch(
        src: np.ndarray, dst: np.ndarray
    ) -> float | None:
        """Estimate the square pitch in pixels from id-adjacent point pairs.

        ``src`` is (n, 2) of (local_col, local_row); ``dst`` is (n, 2) of (x, y).
        id-adjacent pairs are those whose (col, row) differ by exactly 1 in one
        axis and are equal in the other. Returns the median Euclidean pixel
        distance over those pairs, or None if no adjacent pair exists.
        """
        if len(src) < 2:
            return None
        # Build pairwise adjacency by sorting on each axis.
        adj_dists = []
        n = len(src)
        # O(n^2) is fine for n <= ~50 points per face.
        for i in range(n):
            for j in range(i + 1, n):
                dc = src[j, 0] - src[i, 0]
                dr = src[j, 1] - src[i, 1]
                if (abs(dc) == 1 and dr == 0) or (abs(dr) == 1 and dc == 0):
                    adj_dists.append(float(np.linalg.norm(dst[j] - dst[i])))
        if not adj_dists:
            return None
        return float(np.median(adj_dists))

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
        if self.plane_consistency_gate and self.face_reassignment:  # Reassign dropped points to their true face (opt-in, requires the gate).
            keys, image_points = self._run_face_reassignment(keys, image_points, image_shape=np.asarray(image).shape if image is not None else None)
        elif self.plane_consistency_gate:  # Drop geometrically inconsistent merged face points (opt-in gate).
            keys, image_points = self._run_plane_consistency_gate(keys, image_points)
        if not keys:  # Return no data if the decoded points belong only to guard bands (or were all dropped by the gate).
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









