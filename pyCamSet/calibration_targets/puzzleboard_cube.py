from __future__ import annotations

from io import BytesIO
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import svgwrite

from pyCamSet.calibration_targets.core import AbstractTarget, FaceToShape, ImageDetection
from pyCamSet.calibration_targets.core.abstract_target import (
    EXPORT_SUFFIXES, export_path)
from pyCamSet.calibration_targets.core.parameters import (
    DetectorParameterisation,
    DocumentedParameters,
    Parameter,
    Parameterisation,
)
from pyCamSet.calibration_targets.markers.puzzleboard import (
    PUZZLEBOARD_DETECTOR,
    detect_puzzleboard_image,
)
from pyCamSet.calibration_targets.puzzleboard import _CODE_FIELD, _CODE_SIZE
from pyCamSet.cameras import Camera
from pyCamSet.utils.general_utils import make_4x4h_tform


# These transforms use the same face order as the existing Ccube target: front, right, back, left, top, bottom.
TFORMS = [
    (([2.22144147, 2.22144147, 0.0]), ([-0.5, -0.5, 0.5])),
    (([-1.57079633, 0.0, 0.0]), ([-0.5, -0.5, 0.5])),
    (([-1.20919958, -1.20919958, 1.20919958]), ([0.5, -0.5, 0.5])),
    (([0.0, 2.22144147, -2.22144147]), ([0.5, 0.5, 0.5])),
    (([0.0, 0.0, 1.57079633]), ([0.5, -0.5, -0.5])),
    (([1.20919958, 1.20919958, 1.20919958]), ([-0.5, -0.5, -0.5])),
]

# Faces whose outward normals under TFORMS are anti-parallel. Not
# (face + 3) % 6, which is right only for (2, 5).
OPPOSITE_FACE_MAP = {0: 4, 1: 3, 2: 5, 3: 1, 4: 0, 5: 2}

# Below this effective full-field FOV, reassignment is worse than chance
# (measured: 0.28-0.33 against a 0.25 baseline, versus above 0.76 in the
# 20-57 degree range), so the gate drops the cluster instead of firing.
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


CODE_LAYOUT_VERSION = "puzzle-cube-v1"
FACE_COUNT = 6
FACE_GRID_COLUMNS = 3
FACE_GRID_ROWS = 2
FACE_WINDOW_GAP = 0  # Tile adjacent windows directly; their half-open ranges do not overlap.
MAX_FACE_SQUARES = (
    _CODE_SIZE - (FACE_GRID_COLUMNS - 1) * FACE_WINDOW_GAP
) // FACE_GRID_COLUMNS


class PuzzleBoardCubeDetection(DetectorParameterisation):
    """
    What this cube does with what the detector hands back.

    Not the PuzzleBoard detector's settings: these are the two optional
    stages ``find_in_image`` runs over the decoded points, deciding which
    face each one belongs to.  Both are off by default.

    Their thresholds are in units of square pitch, and the numbers below
    come from the round-2 synthetic validation rather than from taste.
    """

    name = "PuzzleBoardCube"

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return (
            Parameter(
                key="plane_consistency_gate", label="Plane consistency gate",
                default=False, dtype="bool", priority="A",
                concept="a two-stage RANSAC homography check per "
                        "face, dropping points not consistent with the "
                        "face's majority plane. Detection: catches two "
                        "physical regions of the cube decoded into one face "
                        "window, which is geometrically impossible. "
                        "Calibration: a merged face is a set of keys in the "
                        "wrong place, which the solve cannot see as wrong.",
                range_text="off / on",
                range_source="Off by default until validated against the "
                             "full component survey",
                suggested="off"),
            Parameter(
                key="plane_gate_contam_squares", label="Contamination threshold",
                default=2.0, dtype="float", tunable=True, settable=True,
                minimum=0.5, maximum=10.0, step=0.1, decimals=3,
                search_order=1, priority="A",
                concept="stage-1 trigger. A face is contaminated "
                        "only if enough of its points sit further than this "
                        "from the all-points homography.",
                range_text="Above the max residual of clean single planes "
                           "(0.69-1.50 squares), below the median residual "
                           "of contaminated merges (1.3-2.2)",
                range_source="Round-2 synthetic validation",
                suggested="2.0"),
            Parameter(
                key="plane_gate_min_contam_frac", label="Contaminated fraction",
                default=0.25, dtype="float", tunable=True, settable=True,
                minimum=0.0, maximum=1.0, step=0.05, decimals=3,
                search_order=2, priority="A",
                concept="how much of a face must be beyond the "
                        "contamination threshold before stage 2 runs.",
                range_text="Clean images have 0% above 2.0 squares; "
                           "contaminated ones have 25-64%",
                range_source="Round-2 synthetic validation",
                suggested="0.25"),
            Parameter(
                key="plane_gate_inlier_squares", label="RANSAC inlier threshold",
                default=0.5, dtype="float", tunable=True, settable=True,
                minimum=0.05, maximum=3.0, step=0.05, decimals=3,
                search_order=3, priority="A",
                concept="stage-2 RANSAC inlier threshold. "
                        "Detection: loose enough and foreign-plane points "
                        "fit the wrong cluster, so max-count RANSAC picks a "
                        "larger wrong model over the correct one.",
                range_text="A true plane's inliers fit at 0.02-0.45 squares "
                           "after RANSAC; at 1.0 a wrong 30-point model beat "
                           "the correct 24-point one",
                range_source="Round-2 synthetic validation",
                suggested="0.5"),
            Parameter(
                key="plane_gate_min_points", label="Minimum face population",
                default=8, dtype="int", tunable=True, settable=True,
                minimum=4, maximum=200, step=1, search_order=4, priority="B",
                concept="the fewest points a face needs before the "
                        "gate will judge it. Below this a homography is not "
                        "worth fitting.",
                range_text="At least 4, which is what a homography needs",
                range_source="Estimated by us",
                suggested="8"),
            Parameter(
                key="plane_gate_ransac_iters", label="RANSAC iterations",
                default=200, dtype="int", tunable=True, settable=True,
                minimum=10, maximum=5000, step=10, search_order=5, priority="B",
                concept="how many models stage 2 samples. More is "
                        "slower and more likely to find the true plane.",
                range_text="Positive; the cost is linear in this",
                range_source="Estimated by us",
                suggested="200"),
            Parameter(
                key="plane_gate_random_state", label="RANSAC seed",
                default=0, dtype="int", settable=True,
                minimum=0, maximum=2 ** 31 - 1, step=1, priority="C",
                concept="the seed the sampling uses, so that a "
                        "detection repeats exactly. Not something to search "
                        "over: a study sweeping it would be optimising "
                        "which noise it happened to like.",
                range_text="Any non-negative integer",
                range_source="Estimated by us",
                suggested="0"),
            Parameter(
                key="face_reassignment", label="Face reassignment",
                default=False, dtype="bool", priority="A",
                concept="rather than dropping a contaminated "
                        "face's minority cluster, work out which co-visible "
                        "face it belongs to and relabel it. Needs the plane "
                        "consistency gate, which is what finds the cluster.",
                range_text="off / on",
                range_source="Round-2 synthetic validation: raw 60-80% "
                             "accuracy across 0-2px noise and grids 4-8",
                suggested="off"),
            Parameter(
                key="face_reassignment_confidence", label="Reassignment confidence",
                default=3.0, dtype="float", tunable=True, settable=True,
                minimum=1.0, maximum=20.0, step=0.5, decimals=3,
                search_order=6, priority="A",
                concept="how much better the best face must fit "
                        "than the second best -- a ratio of reprojection "
                        "errors -- before the cluster is relabelled rather "
                        "than dropped. Detection: a confident wrong "
                        "relabelling is worse than a drop.",
                range_text="Above 3.0 it was right on 93.3% of the ~48% of "
                           "trials it fired on; above 8.0, 98.4% of ~43%",
                range_source="Round-2 synthetic validation",
                suggested="3.0"),
            Parameter(
                key="face_reassignment_intrinsics_fx", label="Assumed focal length (px)",
                default="", dtype="float", settable=True, drop_if_none=True,
                priority="C",
                concept="the generic focal length the tie-breaker's "
                        "PnP assumes. Left empty it is derived from the "
                        "image width. NEVER take this from a calibration: "
                        "a calibration's output must not be an input to the "
                        "detection it was calibrated from.",
                range_text="Empty, or pixels",
                range_source="Default is image_width * 1.4, a moderate "
                             "wide-ish lens",
                suggested="empty"),
            Parameter(
                key="face_reassignment_intrinsics_cx", label="Assumed centre x (px)",
                default="", dtype="float", settable=True, drop_if_none=True,
                priority="C",
                concept="the principal point the tie-breaker's PnP "
                        "assumes. Left empty it is the image centre.",
                range_text="Empty, or pixels",
                range_source="Default is the image centre",
                suggested="empty"),
            Parameter(
                key="face_reassignment_intrinsics_cy", label="Assumed centre y (px)",
                default="", dtype="float", settable=True, drop_if_none=True,
                priority="C",
                concept="the principal point the tie-breaker's PnP "
                        "assumes. Left empty it is the image centre.",
                range_text="Empty, or pixels",
                range_source="Default is the image centre",
                suggested="empty"),
        )

    def validate(self, values: dict) -> list[str]:
        """Reassignment needs the gate that finds the cluster to reassign."""
        if values.get("face_reassignment") and not values.get("plane_consistency_gate"):
            return ["face_reassignment needs plane_consistency_gate: the "
                    "gate is what separates the cluster it relabels."]
        return []


class PuzzleBoardCube(AbstractTarget):
    """Define a deterministic six-face PuzzleBoard calibration target."""

    DETECTOR_BACKENDS = {"puzzle_board": PUZZLEBOARD_DETECTOR}

    @classmethod
    def construction_parameters(cls, backend: str | None = None) -> Parameterisation:
        """What decides the square grid and physical size of a cube."""
        return DocumentedParameters(cls.__init__, "n_points", "length")

    @classmethod
    def own_detector_parameters(cls) -> DetectorParameterisation:
        return PuzzleBoardCubeDetection()

    @classmethod
    def export_parameters(cls) -> Parameterisation:
        """How a PuzzleBoard cube net is drawn, which is not what it is."""
        return DocumentedParameters(
            cls.save_printable, "border_width", "draw_cut_outline",
            "draw_face_ids", "dpi")

    @classmethod
    def printable_name(cls, values: dict, kind: str = "svg") -> str:
        return (f"pcube_{int(values['n_points'])}squares_"
                f"{float(values['length']):g}mm{EXPORT_SUFFIXES[kind]}")

    def save_printable(self, path, kind: str = "svg", border_width: float = 10.0,
                       draw_cut_outline: bool = True, draw_face_ids: bool = True,
                       dpi: int = 300) -> Path:
        """
        Write this cube as a net to print.

        :param path: where to write it
        :param kind: one of :data:`EXPORT_KINDS`
        :param border_width: Net border (mm) -- the margin drawn around the
            folded net. Suggested: 10.
        :param draw_cut_outline: Draw cut outline -- an outline to cut the
            net out along. Suggested: on.
        :param draw_face_ids: Draw face numbers -- a number on each face,
            for folding it the right way up. Suggested: on.
        :param dpi: Raster DPI -- the resolution a raster PDF is rendered
            at. Suggested: 300-600.
        :raises ValueError: for a format a cube cannot be written as
        """
        if kind == "svg":
            return self.save_to_svg(
                path, border_width=border_width,
                draw_cut_outline=draw_cut_outline, draw_face_ids=draw_face_ids)
        if kind in ("pdf_vector", "pdf_raster"):
            return self.save_to_pdf(
                path, data_format="vector" if kind == "pdf_vector" else "raster",
                dpi=int(dpi), border_width=border_width,
                draw_cut_outline=draw_cut_outline, draw_face_ids=draw_face_ids)
        raise ValueError(f"A PuzzleBoardCube cannot be written as {kind!r}.")

    def __init__(
        self,
        n_points: int = 20,
        length: float = 200.0,
        detection_options: dict | None = None,
    ):
        """Initialise a cube whose six faces use disjoint windows of the periodic code.

        :param n_points: Squares per face -- PuzzleBoard squares along one
            edge of one of the cube's six faces. Each face is a separate
            window of the periodic code. Suggested: 10-30.
        :param length: Cube edge (mm) -- the printed edge length of the
            cube, in millimetres. Suggested: 100-300.
        :param detection_options: what the detector is told, by the keys
            :meth:`detector_parameterisation` describes.

        plane_consistency_gate (default False, opt-in): when True, ``find_in_image``
        runs a two-stage RANSAC homography consistency check per face and drops
        points that are not geometrically consistent with the face's majority plane.
        Designed to catch geometrically impossible merged point sets (two physical
        regions of the cube incorrectly decoded into the same face window). Default
        OFF until validated against the full component survey.

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
        never force-assigned). See ``_run_face_reassignment``.

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
        super().__init__(inputs=locals())
        self.n_points = int(n_points)
        self.length = float(length)
        self._validate_dimensions()
        self.square_size = self.length / self.n_points
        # What find_in_image does with the decoded points, as
        # PuzzleBoardCubeDetection resolved it.  The assumed intrinsics
        # stay None when unset: the standing constraint is that a
        # calibration's own output is never an input to the detection it
        # came from, so they are derived from the image at detection time.
        options = self.detection_options
        self.plane_consistency_gate = options["plane_consistency_gate"]
        self.plane_gate_inlier_squares = options["plane_gate_inlier_squares"]
        self.plane_gate_contam_squares = options["plane_gate_contam_squares"]
        self.plane_gate_min_contam_frac = options["plane_gate_min_contam_frac"]
        self.plane_gate_min_points = options["plane_gate_min_points"]
        self.plane_gate_ransac_iters = options["plane_gate_ransac_iters"]
        self.plane_gate_random_state = options["plane_gate_random_state"]
        self.face_reassignment = options["face_reassignment"]
        self.face_reassignment_confidence = options["face_reassignment_confidence"]
        self.face_reassignment_intrinsics_fx = options["face_reassignment_intrinsics_fx"]
        self.face_reassignment_intrinsics_cx = options["face_reassignment_intrinsics_cx"]
        self.face_reassignment_intrinsics_cy = options["face_reassignment_intrinsics_cy"]
        self.min_width = options["min_width"]
        self.layout_version = CODE_LAYOUT_VERSION
        self.face_origins = self.face_origins_for_size(self.n_points)
        self.face_length = self.length / 1000.0
        self.base_face = np.array(
            [[0.0, self.face_length, 0.0], [self.face_length, self.face_length, 0.0],
             [self.face_length, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        self.faceData = FaceToShape(
            face_local_coords=self._make_local_face_points(),
            face_transforms=[make_4x4h_tform(*tform) for tform in TFORMS],
            scale_factor=self.face_length,
        )
        self.point_data = self.faceData.point_data
        self._process_data()

    @classmethod
    def face_origins_for_size(cls, n_points: int) -> tuple[tuple[int, int], ...]:
        """Return deterministic row/column origins for the six disjoint face windows."""
        size = int(n_points)
        if size < 2:
            raise ValueError("n_points must be at least 2.")
        if size > MAX_FACE_SQUARES:
            raise ValueError(
                f"n_points must not exceed {MAX_FACE_SQUARES} for {CODE_LAYOUT_VERSION}; "
                f"three face windows across must fit inside the {_CODE_SIZE}x{_CODE_SIZE} code field."
            )
        step = size + FACE_WINDOW_GAP
        origins = tuple(
            (column * step, row * step)
            for row in range(FACE_GRID_ROWS)
            for column in range(FACE_GRID_COLUMNS)
        )
        if any(x + size > _CODE_SIZE or y + size > _CODE_SIZE for x, y in origins):
            raise ValueError("The deterministic face layout exceeds the PuzzleBoard code field.")
        return origins

    def _validate_dimensions(self) -> None:
        """Validate physical and detector parameters."""
        if self.n_points < 2 or self.n_points > MAX_FACE_SQUARES:
            raise ValueError(f"n_points must be between 2 and {MAX_FACE_SQUARES}.")
        if self.length <= 0.0:
            raise ValueError("length must be greater than zero.")

    def _make_local_face_points(self) -> np.ndarray:
        """Create the six identical local point grids before cube-face transforms."""
        size = self.n_points
        side_m = self.face_length
        points = np.zeros((FACE_COUNT, size * size, 3), dtype=np.float64)
        for face in range(FACE_COUNT):
            for row in range(size):
                for column in range(size):
                    point_id = row * size + column
                    # The detected feature for code (row, column) is the corner
                    # where four squares meet.  Squares are centred on integer
                    # code positions and span half a square each way, so that
                    # corner is half a square in from the integer position --
                    # which also centres the lattice on the face.
                    points[face, point_id, 0] = (column + 0.5) * side_m / size
                    points[face, point_id, 1] = (row + 0.5) * side_m / size
        return points

    @staticmethod
    def _apply_affine(points_xy: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Apply a 3x3 affine matrix to an array of x/y points."""
        homogeneous = np.c_[points_xy, np.ones((len(points_xy), 1), dtype=float)]
        return (affine @ homogeneous.T).T[:, :2]

    def _net_affine_for_face(self, face_index: int) -> np.ndarray:
        """Convert one row/column net transform to an x/y metre transform."""
        face_form = np.asarray(NET_FORMS[face_index], dtype=float)
        permute = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        affine = permute @ face_form @ permute
        affine[:2, 2] *= self.face_length
        return affine

    def _face_rectangles(self, face_index: int) -> list[np.ndarray]:
        """Return black checkerboard polygons for one face in local metres."""
        size = self.n_points
        side_m = self.face_length
        square_m = self.square_size / 1000.0
        rectangles: list[np.ndarray] = []
        start_x, start_y = self.face_origins[face_index]
        for x in range(size + 1):
            for y in range(size + 1):
                if (x + y + start_x + start_y) % 2 == 1:
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
                if x == size:
                    width = 0.5
                if y == size:
                    height = 0.5
                rectangles.append(np.array(
                    [[xx * square_m, yy * square_m], [(xx + width) * square_m, yy * square_m],
                     [(xx + width) * square_m, (yy + height) * square_m], [xx * square_m, (yy + height) * square_m]],
                    dtype=float,
                ))
        return rectangles

    def _face_circles(self, face_index: int) -> list[tuple[np.ndarray, str]]:
        """Return encoded circle centres and colours for one face in local metres."""
        size = self.n_points
        square_m = self.square_size / 1000.0
        radius_m = square_m / 6.0
        start_x, start_y = self.face_origins[face_index]
        circles: list[tuple[np.ndarray, str]] = []
        for row in range(size):
            for column in range(size - 1):
                value = int(_CODE_FIELD[start_y + row, start_x + column])
                colour = "white" if value & 2 else "black"  # Bit 2 encodes the horizontal edge.
                circles.append((np.array([(column + 1) * square_m, (row + 0.5) * square_m]), colour))
        for row in range(size - 1):
            for column in range(size):
                value = int(_CODE_FIELD[start_y + row, start_x + column])
                colour = "white" if value & 1 else "black"  # Bit 1 encodes the vertical edge.
                circles.append((np.array([(column + 0.5) * square_m, (row + 1) * square_m]), colour))
        return circles

    def _add_face_geometry(
        self,
        drawing: svgwrite.Drawing,
        face_index: int,
        affine: np.ndarray,
        offset_m: np.ndarray,
    ) -> None:
        """Add one transformed face's checkerboard and code circles to an SVG drawing."""
        for rectangle in self._face_rectangles(face_index):
            points_mm = self._apply_affine(rectangle, affine) * 1000.0 + offset_m * 1000.0
            drawing.add(drawing.polygon(points=[tuple(point) for point in points_mm], fill="black", stroke="none"))
        radius_mm = self.square_size / 6.0
        for centre, colour in self._face_circles(face_index):
            point_mm = self._apply_affine(centre.reshape(1, 2), affine)[0] * 1000.0 + offset_m * 1000.0
            drawing.add(drawing.circle(center=(float(point_mm[0]), float(point_mm[1])), r=radius_mm, fill=colour, stroke="none"))

    def _net_bounds(self, border_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the printable net bounds and page offset in metres."""
        outlines: list[np.ndarray] = []
        for face_index in range(FACE_COUNT):
            affine = self._net_affine_for_face(face_index)
            outline = np.array([[0.0, 0.0], [self.face_length, 0.0], [self.face_length, self.face_length], [0.0, self.face_length]])
            outlines.append(self._apply_affine(outline, affine))
        all_points = np.vstack(outlines)
        border_m = float(border_width) / 1000.0
        min_xy = all_points.min(axis=0) - border_m
        max_xy = all_points.max(axis=0) + border_m
        offset_m = -min_xy
        return min_xy, max_xy, offset_m

    def _svg_document(
        self,
        border_width: float = 10.0,
        draw_cut_outline: bool = True,
        draw_face_ids: bool = False,
    ) -> tuple[svgwrite.Drawing, float, float]:
        """Construct a vector SVG of the six-face printable cube net."""
        min_xy, max_xy, offset_m = self._net_bounds(border_width)
        canvas_w_mm = float((max_xy[0] - min_xy[0]) * 1000.0)
        canvas_h_mm = float((max_xy[1] - min_xy[1]) * 1000.0)
        drawing = svgwrite.Drawing(
            size=(f"{canvas_w_mm:.6f}mm", f"{canvas_h_mm:.6f}mm"),
            viewBox=f"0 0 {canvas_w_mm:.6f} {canvas_h_mm:.6f}",
        )
        drawing.add(drawing.rect(insert=(0.0, 0.0), size=(canvas_w_mm, canvas_h_mm), fill="white"))
        for face_index in range(FACE_COUNT):
            affine = self._net_affine_for_face(face_index)
            self._add_face_geometry(drawing, face_index, affine, offset_m)
            if draw_cut_outline:
                outline = np.array([[0.0, 0.0], [self.face_length, 0.0], [self.face_length, self.face_length], [0.0, self.face_length]])
                points_mm = self._apply_affine(outline, affine) * 1000.0 + offset_m * 1000.0
                drawing.add(drawing.polygon(points=[tuple(point) for point in points_mm], fill="none", stroke="black", stroke_width=0.2))
            if draw_face_ids:
                label = np.array([[0.02 * self.face_length, 0.985 * self.face_length]])
                label_mm = self._apply_affine(label, affine)[0] * 1000.0 + offset_m * 1000.0
                label_size_mm = self.face_length * 1000.0 * 0.045
                label_angle_deg = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
                label_text = drawing.text(  # Use viewBox millimetres directly; an additional mm suffix would rescale the text.
                    str(face_index + 1),
                    insert=tuple(label_mm),
                    fill="white",
                    font_size=f"{label_size_mm:.6f}",
                    font_family="Arial",
                    font_weight="bold",
                )
                label_text.rotate(label_angle_deg, center=tuple(label_mm))
                drawing.add(label_text)
        return drawing, canvas_w_mm, canvas_h_mm

    def _face_svg(self, face_index: int) -> str:
        """Create a square SVG texture for one face for 3-D visualisation."""
        side_mm = self.face_length * 1000.0
        drawing = svgwrite.Drawing(
            size=(f"{side_mm:.6f}mm", f"{side_mm:.6f}mm"),
            viewBox=f"0 0 {side_mm:.6f} {side_mm:.6f}",
        )
        drawing.add(drawing.rect(insert=(0.0, 0.0), size=(side_mm, side_mm), fill="white"))
        self._add_face_geometry(drawing, face_index, np.eye(3), np.zeros(2))
        drawing.add(drawing.text(
            str(face_index + 1),
            insert=(float(side_mm * 0.02), float(side_mm * 0.985)),
            fill="white",
            font_size=f"{side_mm * 0.045:.6f}",
            font_family="Arial",
            font_weight="bold",
        ))
        return drawing.tostring()

    def save_to_svg(
        self,
        f_out: Path | str,
        border_width: float = 10.0,
        draw_cut_outline: bool = True,
        draw_face_ids: bool = True,
        suppress_svg_log: bool = False,
    ) -> Path:
        """Save the deterministic cube net as a physically sized vector SVG."""
        f_out = export_path(f_out, ".svg")
        drawing, _, _ = self._svg_document(border_width, draw_cut_outline, draw_face_ids)
        svg_text = drawing.tostring()
        with open(f_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg_text)
            fh.flush()
        if (not f_out.exists()) or f_out.stat().st_size == 0:


            raise IOError(f"SVG write failed: {f_out}")
        if not suppress_svg_log:
            logging.info("Saved PuzzleBoard cube SVG: %s", f_out)
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
        """Save the cube net as a raster or vector PDF."""
        f_out = export_path(f_out, ".pdf")
        drawing, canvas_w_mm, canvas_h_mm = self._svg_document(border_width, draw_cut_outline, draw_face_ids)
        svg_bytes = drawing.tostring().encode("utf-8")
        from pyCamSet.utils.cairo_dll_helper import cairosvg_or_explain
        cairosvg = cairosvg_or_explain()
        if data_format == "vector":
            cairosvg.svg2pdf(bytestring=svg_bytes, write_to=str(f_out))
            logging.info("Saved PuzzleBoard cube Vector PDF: %s", f_out)
            return f_out
        if data_format != "raster":


            raise ValueError("data_format must be one of: raster, vector")
        png = cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=max(1, int(round(canvas_w_mm / 25.4 * dpi))),
            output_height=max(1, int(round(canvas_h_mm / 25.4 * dpi))),
        )
        with Image.open(BytesIO(png)) as image:
            image.convert("RGB").save(f_out, resolution=float(dpi))
        logging.info("Saved PuzzleBoard cube Raster PDF: %s", f_out)
        return f_out

    def plot(self, return_scene: bool = False, draw_res: tuple[int, int] = (800, 800)):
        """Visualise the six textured faces in a 3-D pyVista scene."""
        from pyCamSet.utils.cairo_dll_helper import cairosvg_or_explain
        cairosvg = cairosvg_or_explain()
        textures: list[np.ndarray] = []
        for face_index in range(FACE_COUNT):
            png = cairosvg.svg2png(
                bytestring=self._face_svg(face_index).encode("utf-8"),
                output_width=int(draw_res[0]),
                output_height=int(draw_res[1]),
            )
            with Image.open(BytesIO(png)) as image:
                textures.append(np.asarray(image.convert("RGB")))
        scene = self.faceData.draw_meshes(self.base_face, textures, return_scene=return_scene)
        if return_scene:
            return scene
        return None

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

        size = self.n_points
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
        ~13deg (fx > ~2000 on a 500px-wide image). When the assumed fx and the
        detection-time image width imply a full FOV below
        ``FACE_REASSIGNMENT_MIN_SAFE_FOV_DEG``, this method refuses to run the
        PnP tie-breaker entirely and instead falls back to the gate's existing
        drop path for every contaminated face (the dropped cluster is dropped,
        never force-assigned). This matches the "never force an assignment the
        signal does not support" design philosophy.
        """
        if not keys:
            return keys, image_points

        size = self.n_points
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
        del camera
        point_ids, point_coords = detect_puzzleboard_image(image, min_width=self.min_width)
        if len(point_ids) == 0:
            return ImageDetection()
        positions = np.asarray(point_ids, dtype=np.int64)  # Detector positions are [row, column].
        coordinates = np.asarray(point_coords, dtype=np.float64)  # Detector image coordinates are [row, column].
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("PuzzleBoard detector positions must have shape (n, 2).")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("PuzzleBoard detector coordinates must have shape (n, 2).")
        keys: list[list[int]] = []
        image_points: list[np.ndarray] = []
        size = self.n_points
        for position, coordinate in zip(positions, coordinates):
            row, column = int(position[0]), int(position[1])  # Preserve the detector's row/column convention.
            matches = []
            for face_index, (origin_x, origin_y) in enumerate(self.face_origins):
                local_column = column - origin_x
                local_row = row - origin_y
                if 0 <= local_column < size and 0 <= local_row < size:
                    matches.append((face_index, local_row, local_column))
            if len(matches) != 1:  # Reject guard-band or ambiguous points rather than assigning the wrong face.
                continue
            face_index, local_row, local_column = matches[0]
            keys.append([face_index, local_row * size + local_column])
            image_points.append(coordinate[::-1])  # Convert detector [row, column] to pyCamSet [x, y].
        if self.plane_consistency_gate and self.face_reassignment:  # Reassign dropped points to their true face (opt-in, requires the gate).
            keys, image_points = self._run_face_reassignment(keys, image_points, image_shape=np.asarray(image).shape if image is not None else None)
        elif self.plane_consistency_gate:  # Drop geometrically inconsistent merged face points (opt-in gate).
            keys, image_points = self._run_plane_consistency_gate(keys, image_points)
        if not keys:
            return ImageDetection()
        image_points_array = np.asarray(image_points, dtype=np.float64)
        if draw:
            display_im = np.asarray(image).copy()
            for point in image_points_array:
                cv2.circle(display_im, (int(round(point[0])), int(round(point[1]))), 3, (0, 0, 255), 1)
            cv2.imshow("PuzzleBoard cube detections", display_im)
            cv2.waitKey(wait_len)
        return ImageDetection(np.asarray(keys, dtype=np.int64), image_points_array)

