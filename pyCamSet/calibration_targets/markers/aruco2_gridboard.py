"""
aruco2's ``GridBoard`` detector -- the ChArUco2 design.

Every square of the board carries an ArUco marker (a standard marker on a
black square, an inverted one on a white square), giving an N x M board
(N x M markers) and (N+1) x (M+1) observable intersection corners including
the board's own border. The design is described in
https://www.sciencedirect.com/science/article/pii/S2352711026003249.

``aruco2.detect_grid_board`` hands back a populated ``GridBoard`` object.
Its Python API exposes the found corners only indirectly, through
``aruco2.get_solve_pnp_points(board, marker_size)``, which returns matched
``(object_points, image_points)`` arrays for the corners it actually found.
Mapping a detection back to pyCamSet's stable per-corner index (matching
:data:`ChArUco2.point_data`'s row order) therefore has to go through the
object points, not through anything positional in ``markers``. Every
returned object point lies on the ``(grid_width+1) x (grid_height+1)``
intersection lattice at ``(col, row) * marker_size``, so::

    row = round(object_point[1] / marker_size)
    col = round(object_point[0] / marker_size)
    gid = row * (grid_width + 1) + col

recovers the same row-major index :func:`pyCamSet.calibration_targets
.charuco2.target.ChArUco2`'s own ``point_data`` is built with. This mapping
was verified empirically against a full board, a board with one quadrant
occluded, a 90-degree-rotated board, and a perspective-warped board: in
every case the surviving corners' recovered ``gid``s pointed at the same
physical 3-D point the full, undistorted board assigned them, with
sub-pixel image-point reprojection error.
"""

from __future__ import annotations

import logging

import numpy as np

from pyCamSet.calibration_targets.markers.aruco2 import (
    ARUCO2_AVAILABLE,
    _require_aruco2,
    _as_uint8_image,
)
from pyCamSet.calibration_targets.markers.backend_registry import (
    ARUCO2_BACKEND,
    marker_backend_available,
)
from pyCamSet.calibration_targets.core.parameters import (
    Parameter,
    DetectorParameterisation,
)

_LOG = logging.getLogger(__name__)

# Reuse the same lazy-import guard as the aruco2 marker backend
# (pyCamSet.calibration_targets.markers.aruco2, imported above): this module
# still needs its own local ``aruco2`` reference to call
# detect_grid_board/get_solve_pnp_points/etc, but must import cleanly even
# when the package is not installed.
try:
    import aruco2  # noqa: F401
except (ImportError, OSError):  # pragma: no cover - exercised by the mocked gate check
    aruco2 = None  # type: ignore[assignment]


def detect_grid_board_corners(
    image, grid_size: tuple[int, int], dict_int: int, marker_size: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Detect a ChArUco2 grid board and map it to pyCamSet's stable corner ids.

    :param image: the image to detect in, uint8 or an integral 0..255 float.
    :param grid_size: ``(num_squares_x, num_squares_y)``, in markers.
    :param dict_int: the aruco2 dictionary id every square is printed with.
    :param marker_size: the physical pitch between two adjacent markers'
        object points (the printed square size), in the same units
        :data:`ChArUco2.point_data` is stored in.
    :return: ``(corner_ids, image_points)`` -- ``corner_ids`` a 1D int array
        indexing :data:`ChArUco2.point_data`'s rows, ``image_points`` an
        ``(N, 2)`` float array of where each one was found -- or
        ``(None, None)`` when nothing was detected.
    """
    _require_aruco2()
    img = _as_uint8_image(image)
    grid_w, grid_h = int(grid_size[0]), int(grid_size[1])
    found, board = aruco2.detect_grid_board(img, (grid_w, grid_h), int(dict_int))
    if not found or not board.markers:
        return None, None

    obj_points, img_points = aruco2.get_solve_pnp_points(
        board, marker_size=float(marker_size))
    obj_points = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
    img_points = np.asarray(img_points, dtype=np.float64).reshape(-1, 2)
    if obj_points.shape[0] == 0:
        return None, None

    # Every returned object point lies exactly on the (grid_w+1) x
    # (grid_h+1) intersection lattice at (col, row) * marker_size; recover
    # the row-major index (== the corner's global id in aruco2's own
    # numbering, which pyCamSet.point_data is built to match) by rounding
    # rather than trusting float equality.
    col = np.round(obj_points[:, 0] / float(marker_size)).astype(np.int64)
    row = np.round(obj_points[:, 1] / float(marker_size)).astype(np.int64)
    corner_ids = row * (grid_w + 1) + col
    return corner_ids, img_points.astype(np.float64)


def render_grid_board_image(
    grid_size: tuple[int, int], dict_int: int, bit_size: int,
) -> np.ndarray:
    """The exact raster aruco2 both prints and detects a grid board from."""
    _require_aruco2()
    grid_w, grid_h = int(grid_size[0]), int(grid_size[1])
    return aruco2.get_grid_board_image((grid_w, grid_h), int(dict_int), int(bit_size))


def dictionary_marker_bits(dict_int: int) -> int:
    """How many bits wide/tall one marker's payload is, for this dictionary."""
    _require_aruco2()
    return int(aruco2.get_predefined_dictionary(int(dict_int)).marker_size)


class ChArUco2Detector(DetectorParameterisation):
    """
    aruco2's grid-board detector, which takes no settings.

    ``aruco2.detect_grid_board`` is given an image, a board size and a
    dictionary and nothing else -- so there is no ``DetectionParameters``-
    style control here for a form to show or a study to sweep. It runs a
    fixed internal detection pipeline with no exposed tuning.
    """

    name = ARUCO2_BACKEND

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return ()

    def unavailable_reason(self, values: dict | None = None) -> str | None:
        if marker_backend_available(ARUCO2_BACKEND):
            return None
        return (
            "ChArUco2 detection requires the 'aruco2' package, which is not "
            "installed. Install it with `pip install aruco2` -- unlike "
            "ChArUco/Ccube, ChArUco2 has no ArUco 1 (OpenCV) fallback to "
            "switch to.")


#: Shared rather than built per target: it describes nothing per instance.
CHARUCO2_DETECTOR = ChArUco2Detector()
