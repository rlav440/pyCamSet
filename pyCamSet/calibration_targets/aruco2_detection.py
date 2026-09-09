'''
Purpose: ArUco2 (aruco2 package) marker backend for pyCamSet ChArUco/Ccube
         targets. Implements plan v4 decisions D3 (dictionary resolution) and
         D4/D5 (marker detection + ChArUco corner interpolation) for the
         aruco2 backend; the aruco1 (OpenCV) path is untouched.
Status: Active. Batch B1 of the aruco1/aruco2 backend work (plan v4).
Future: If aruco2 gains native ChArUco support, this module may shrink to a
        thin adapter; the cross-marker disagreement metric stays as the
        legacy-pattern discriminator.
'''

from __future__ import annotations

import logging

import cv2
import numpy as np

from pyCamSet.calibration_targets.backend_registry import validate_marker_backend

_LOG = logging.getLogger(__name__)

# Module-level lazy aruco2 import guard (D3): the module must import cleanly
# even when aruco2 is not installed. ARUCO2_AVAILABLE is the single source of
# truth for availability checks in the target classes.
ARUCO2_AVAILABLE: bool = False
try:
    import aruco2  # noqa: F401  (used lazily through the module reference)
    ARUCO2_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover - exercised by the mocked gate check
    aruco2 = None  # type: ignore[assignment]


def _require_aruco2() -> None:
    """Raise an actionable ImportError when the aruco2 package is missing."""
    if not ARUCO2_AVAILABLE:
        raise ImportError(
            "marker_backend='aruco2' requires the 'aruco2' package, which is not "
            "installed. Install it with: pip install aruco2"
        )


def _as_uint8_image(image) -> np.ndarray:
    """Return a contiguous uint8 image, rejecting unsafe conversions."""
    img = np.asarray(image)
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)
    convertible = (
        np.issubdtype(img.dtype, np.floating)
        and img.size > 0
        and np.all(np.isfinite(img))
        and np.allclose(img, np.round(img))
        and float(np.min(img)) >= 0.0
        and float(np.max(img)) <= 255.0
    )
    if not convertible:
        raise ValueError(
            "aruco2 detection requires a uint8 image or an integral floating-point image "
            f"in the range 0..255; got dtype {img.dtype}."
        )
    return np.ascontiguousarray(img.astype(np.uint8))


def resolve_dictionary(a_dict, marker_backend: str = "aruco1") -> cv2.aruco.Dictionary:
    """D3: resolve a dictionary int to a cv2.aruco.Dictionary for the backend.

    aruco1: reject the ALVAR alias ints {22, 23} (OpenCV silently maps them to
    DICT_4X4_50) and otherwise keep the current getPredefinedDictionary(int)
    path. aruco2: build a cv2 Dictionary from the aruco2 package's bytes so
    detection and rendering agree.
    """
    validate_marker_backend(marker_backend)
    if isinstance(a_dict, cv2.aruco.Dictionary):
        # FIX 1 (R1): the pre-existing Ccube API accepts a resolved
        # cv2.aruco.Dictionary object; split_aruco_dictionary handles both
        # ints and Dictionary objects, so pass the object through unchanged.
        if marker_backend == "aruco2":
            raise ValueError(
                "marker_backend='aruco2' requires a dictionary int (e.g. "
                "cv2.aruco.DICT_4X4_1000) so the aruco2 bytes can be resolved."
            )
        return a_dict
    dict_int = int(a_dict)
    if marker_backend == "aruco1":
        if dict_int in (22, 23):
            raise ValueError(
                f"aruco dictionary int {dict_int} is an ALVAR alias that OpenCV "
                "silently maps to DICT_4X4_50; use marker_backend='aruco2' for "
                "ALVAR dictionaries."
            )
        return cv2.aruco.getPredefinedDictionary(dict_int)
    if marker_backend == "aruco2":
        _require_aruco2()
        a2_dict = aruco2.get_predefined_dictionary(dict_int)
        cv2_dict = cv2.aruco.Dictionary(
            np.asarray(a2_dict.bytes_list, dtype=np.uint8).copy(),
            int(a2_dict.marker_size),
        )
        cv2_dict.maxCorrectionBits = int(a2_dict.max_correction_bits)
        return cv2_dict
    raise ValueError(
        f"marker_backend must be 'aruco1' or 'aruco2', got {marker_backend!r}"
    )


def detect_markers(image, dict_int):
    """Run aruco2 detection on a uint8 image.

    :param image: the image to detect in (uint8, 1 or 3 channel).
    :param dict_int: the parent dictionary int (global id space).
    :return: list of (marker_id, corners (4,2) float32) tuples.
    """
    _require_aruco2()
    img = _as_uint8_image(image)
    markers = aruco2.detect_fiducial_markers(img, int(dict_int))
    return [(int(m.id), np.asarray(m.corners, dtype=np.float32)) for m in markers]


def detect_charuco_corners(image, board, dict_int, warn_legacy=None):
    """D4: detect markers with aruco2 and interpolate ChArUco corners.

    :param image: the (uint8) image to detect in.
    :param board: a cv2.aruco.CharucoBoard defining the target geometry.
    :param dict_int: the dictionary int used for detection.
    :param warn_legacy: optional zero-arg callable fired once when the best
        legacy-pattern run still exceeds the disagreement threshold.
    :return: (corner_ids 1D int array, pts (N,2) float32) or (None, None).
    """
    image = _as_uint8_image(image)
    markers = detect_markers(image, dict_int)
    return interpolate_board_corners(image, board, markers, warn_legacy=warn_legacy)


def interpolate_board_corners(image, board, markers, warn_legacy=None):
    """D4 steps 3-10: interpolate chessboard corners from detected markers.

    :param image: the (uint8) image the markers were detected in.
    :param board: a cv2.aruco.CharucoBoard whose geometry defines the corners.
    :param markers: list of (marker_id, corners (4,2) float32) with ids in the
        board's local id space (Ccube callers remap global ids first).
    :param warn_legacy: optional zero-arg callable fired once when the best
        legacy-pattern run still exceeds the disagreement threshold.
    :return: (corner_ids 1D int array, pts (N,2) float32) or (None, None).
    """
    image = _as_uint8_image(image)
    square_len = float(board.getSquareLength())
    marker_len = float(board.getMarkerLength())
    inset = (square_len - marker_len) / 2.0
    h, w = image.shape[:2]

    def _interpolate_with_board(b):
        """D4 steps 2-7 against one board geometry.

        Returns (ids, pts, max_disagreement, estimates) where estimates maps
        corner id -> list of (marker_id, image_pt, marker_corners) so the
        refinement step can find the contributing markers.
        """
        board_ids = np.asarray(b.getIds()).reshape(-1).astype(int)
        board_id_set = set(int(i) for i in board_ids)
        chess = np.asarray(b.getChessboardCorners(), dtype=np.float64).squeeze()
        if chess.ndim == 3:
            chess = chess.reshape(-1, chess.shape[-1])
        chess_xy = chess[:, :2]

        estimates = {}
        for mid, corners in markers:
            mid = int(mid)
            # step 2: keep only markers whose id is on this board; stray ids
            # are skipped (prevents IndexError on foreign markers).
            if mid not in board_id_set:
                _LOG.debug("aruco2: skipping marker id %d not on board", mid)
                continue
            corners = np.asarray(corners, dtype=np.float32)
            if corners.shape != (4, 2):
                _LOG.debug("aruco2: skipping marker id %d with shape %s", mid, corners.shape)
                continue
            # skip markers with non-finite or zero-area quads, or corners
            # outside the image (R2-P1-2: the old |H(obj)-img| check was
            # tautological because H maps its own sources exactly).
            if not np.all(np.isfinite(corners)):
                continue
            if cv2.contourArea(corners) <= 0.0:
                continue
            if (np.any(corners[:, 0] < 0) or np.any(corners[:, 1] < 0)
                    or np.any(corners[:, 0] >= w) or np.any(corners[:, 1] >= h)):
                continue
            # step 3: local quad from the board. getObjPoints() is a TUPLE of
            # (4,3) arrays; index with the int row position, never fancy-index.
            row = np.where(board_ids == mid)[0]
            if len(row) == 0:
                continue
            idx = int(row[0])
            obj_quad = np.asarray(b.getObjPoints()[idx], dtype=np.float64)[:, :2]
            # expand the marker quad to the full square by the inset, diagonally
            # from the quad centre (axis-aligned in board coords, so this is the
            # exact square-corner expansion; valid for both legacy patterns).
            expanded = obj_quad.copy()
            expanded[0] += [-inset, -inset]
            expanded[1] += [inset, -inset]
            expanded[2] += [inset, inset]
            expanded[3] += [-inset, inset]
            # step 4: match expanded square corners to chessboard corners;
            # drop unmatched (board-boundary) corners.
            matched = []
            for corner in expanded:
                d = np.linalg.norm(chess_xy - corner, axis=1)
                j = int(np.argmin(d))
                if d[j] < 1e-6:
                    matched.append(j)
            if not matched:
                continue
            # step 5: local homography from the marker's own corners.
            marker_obj = np.asarray(obj_quad, dtype=np.float32).reshape(4, 1, 2)
            img_corners = corners.reshape(4, 1, 2)
            H = cv2.getPerspectiveTransform(marker_obj, img_corners)
            # step 6: map the matched chessboard corners through H.
            src = np.asarray(chess_xy[matched], dtype=np.float32).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(src, H).reshape(-1, 2)
            for cid, pt in zip(matched, mapped):
                estimates.setdefault(cid, []).append((mid, pt, corners))

        if not estimates:
            return None, None, 0.0, {}

        # step 7: average duplicate corner ids across markers; record the
        # per-corner contributor lists for the disagreement metric.
        ids = sorted(estimates.keys())
        pts = np.empty((len(ids), 2), dtype=np.float32)
        max_disagreement = 0.0
        for k, cid in enumerate(ids):
            ests = estimates[cid]
            pts[k] = np.mean([e[1] for e in ests], axis=0)
            if len(ests) >= 2:
                for i in range(len(ests)):
                    for j in range(i + 1, len(ests)):
                        max_disagreement = max(
                            max_disagreement,
                            float(np.linalg.norm(ests[i][1] - ests[j][1])),
                        )
        return np.asarray(ids, dtype=np.int32), pts, max_disagreement, estimates

    corner_ids, pts, disagreement, estimates = _interpolate_with_board(board)
    if corner_ids is None:
        return None, None

    # step 8: cross-marker disagreement legacy toggle. Even-row boards only
    # (odd-row boards never toggle - aruco_board.cpp:335). On mismatch, toggle
    # the board flag, re-run interpolation with the SAME markers, and keep the
    # run with the lower disagreement; warn once if the best run still exceeds
    # 5px (the geometry supports no better answer).
    n_y = int(board.getChessboardSize()[1])
    if n_y % 2 == 0 and disagreement > 5.0:
        board.setLegacyPattern(not board.getLegacyPattern())
        t_ids, t_pts, t_disagreement, t_estimates = _interpolate_with_board(board)
        # FIX 8(f): only accept the toggled run when it actually produced
        # corners (t_ids is not None); a None toggled run must not be kept.
        if t_ids is not None and t_disagreement < disagreement:
            corner_ids, pts, disagreement, estimates = (
                t_ids, t_pts, t_disagreement, t_estimates,
            )
        else:
            # the original flag was right (or the toggled run found nothing);
            # restore it so the board object stays consistent with the kept run.
            board.setLegacyPattern(not board.getLegacyPattern())
        if disagreement > 5.0 and warn_legacy is not None:
            warn_legacy()

    # step 9: subpixel refinement (belt-and-braces; aruco2 already refines
    # marker corners, but the extrapolated square corners need it).
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # FIX 8(e): cornerSubPix requires uint8; integral float64/float32 inputs
    # (e.g. raw Ccube textures) are converted here too, matching detect_markers.
    if gray.dtype in (np.float32, np.float64) and np.allclose(gray, np.round(gray)):
        gray = gray.astype(np.uint8)
    refined = np.empty_like(pts)
    for k, (cid, pt) in enumerate(zip(corner_ids, pts)):
        # window = max(1, min(10, int(dist)-2)) where dist is the distance to
        # the nearest corner of any CONTRIBUTING marker (charuco_detector.cpp:120).
        dist = float("inf")
        for _mid, _pt, mcorners in estimates[int(cid)]:
            dist = min(dist, float(np.min(np.linalg.norm(mcorners - pt, axis=1))))
        win = max(1, min(10, int(dist) - 2))
        in_pt = np.asarray([[pt[0] - 0.5, pt[1] - 0.5]], dtype=np.float32)
        cv2.cornerSubPix(
            gray,
            in_pt,
            (win, win),
            (-1, -1),
            (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.1),
        )
        new_pt = in_pt[0] + 0.5
        # accept the refined position only if it moved < 2px, else keep the
        # homography point (R1-P0(CV) belt-and-braces contract).
        if float(np.linalg.norm(new_pt - pt)) < 2.0:
            refined[k] = new_pt
        else:
            refined[k] = pt

    # step 10: return (corner_ids 1D, pts (N,2)).
    return corner_ids, refined
