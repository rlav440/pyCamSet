'''
:Purpose: ArUco2 (aruco2 package) marker backend for pyCamSet ChArUco/Ccube
    targets. Implements plan v4 decisions D3 (dictionary resolution) and
    D5 (marker detection) for the aruco2 backend; the aruco1 (OpenCV) path
    is untouched. ChArUco corner interpolation (D4) is an adapter: aruco2's
    own markers (their quads corrected where aruco2's corner refinement
    biases them, see ``_seed_quads``) are handed to a cached
    ``cv2.aruco.CharucoDetector`` per board, so the same native
    interpolation and sub-pixel refinement the aruco1 path uses also reads
    aruco2's markers -- there is no separate
    homography/refinement implementation to keep in step with OpenCV's own.
:Status: Active. aruco1/aruco2 backend work (plan v4), corner-interpolation
    adapter revision; legacy-pattern handling rewritten to the approved
    warn-only policy (no per-frame retry, no ambiguity-discard -- see
    :func:`interpolate_board_corners`'s docstring and
    ``markers/legacy_probe.py``).
:Future: If aruco2 gains native ChArUco support, this module may shrink
    further. A duplicate marker id is dropped before detection reaches
    OpenCV at all (an ambiguous INPUT, never a corrupted one: which of the
    two same-id markers is real cannot be told here). Beyond that, this
    module does not try to guess which marker is "wrong": a round-1 review
    found that an earlier per-marker consistency heuristic could itself
    return confidently wrong, mislabelled corners under a legacy-pattern
    mismatch on a partial view, which is worse than the fail-safe empty
    result it was meant to avoid (see :func:`interpolate_board_corners`'s
    docstring). A frame that the configured pattern rejects outright is
    therefore left as no corners -- fewer corners, never a wrong one.
'''

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from pyCamSet.calibration_targets.markers.backend_registry import validate_marker_backend

_LOG = logging.getLogger(__name__)

from pyCamSet.calibration_targets.markers.backend_registry import (
    ARUCO2_BACKEND,
    marker_backend_available,
)
from pyCamSet.calibration_targets.core.parameters import (
    Parameter,
    DetectorParameterisation,
)
# The corner-interpolation adapter below hands aruco2's markers to
# OpenCV's CharucoDetector, built through the same helper the aruco1
# path uses, so both are configured alike.
from pyCamSet.calibration_targets.markers.aruco_opencv import ARUCO_OPENCV_DETECTOR
from pyCamSet.calibration_targets.markers.legacy_probe import should_warn_legacy_mismatch

# This module must import cleanly without aruco2 installed;
# ARUCO2_AVAILABLE is the single source of truth for that. The two
# failures are kept apart because their fixes differ: ImportError is
# 'not installed', OSError is installed but its native runtime DLLs
# would not load, and _require_aruco2 reports that text.
try:
    import aruco2  # noqa: F401  (used lazily through the module reference)
    _aruco2_importable = True
    _aruco2_import_oserror: OSError | None = None
except ImportError:  # pragma: no cover - exercised by the mocked gate check
    aruco2 = None  # type: ignore[assignment]
    _aruco2_importable = False
    _aruco2_import_oserror = None
except OSError as _err:  # pragma: no cover - exercised by the mocked gate check
    aruco2 = None  # type: ignore[assignment]
    _aruco2_importable = False
    _aruco2_import_oserror = _err

ARUCO2_AVAILABLE: bool = _aruco2_importable


def _require_aruco2() -> None:
    """Raise an actionable ImportError when the aruco2 package is missing or
    could not be loaded."""
    if ARUCO2_AVAILABLE:
        return
    if _aruco2_import_oserror is not None:
        # str(OSError) often already ends in a full stop (as Windows' own
        # DLL-load messages do); strip one trailing stop so the sentence
        # that follows does not read "...found.. This is usually...".
        oserror_text = str(_aruco2_import_oserror).rstrip()
        if oserror_text.endswith("."):
            oserror_text = oserror_text[:-1]
        raise ImportError(
            "marker_backend='aruco2' requires the 'aruco2' package. It was "
            "found, but loading its native libraries failed with: "
            f"{oserror_text}. This is usually a missing Microsoft "
            "Visual C++ runtime (the aruco2 wheel links against CONCRT140.dll, "
            "MSVCP140.dll and VCRUNTIME140.dll) rather than aruco2 itself not "
            "being installed -- install the Visual C++ Redistributable and "
            "retry."
        )
    raise ImportError(
        "marker_backend='aruco2' requires the 'aruco2' package, which is not "
        "installed. It is not published on PyPI: build it from the "
        "third_party/aruco2 submodule, as described under 'Installing the "
        "aruco2 backend' in pyCamSet's CITATION.md."
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


#: Bounds a marker id must fall within to survive the np.int32 cast
#: below: an id outside it raises OverflowError from numpy instead of
#: being skipped the way a foreign id is.
_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)


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


#: How many (board, detector) entries the cache keeps. Generous: a
#: target holds a handful of live boards, so it binds only when
#: something rebuilds targets repeatedly.
_CHARUCO_DETECTOR_CACHE_MAXSIZE = 64

#: One CharucoDetector per board, keyed by id(board) but holding the
#: board too: CPython reuses the address of a collected object, and
#: detectBoard never checks that its detector's board matches the
#: geometry the caller means. The stored board is compared by identity,
#: so a recycled address misses and rebuilds rather than silently
#: reading one board with another's detector.
_CHARUCO_DETECTOR_CACHE: dict[int, tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]] = {}

#: Guards every read-check-insert-evict sequence on the cache. The
#: cache is process-local, but a caller driving this from OS threads
#: can race the eviction: two threads picking the same oldest key
#: (KeyError), or a mutation landing mid-iteration (RuntimeError).
#: Both were reproduced. The hit-dominated path pays only an
#: uncontended acquisition.
_CHARUCO_DETECTOR_CACHE_LOCK = threading.Lock()


def _charuco_detector_for(board) -> cv2.aruco.CharucoDetector:
    """The ``cv2.aruco.CharucoDetector`` for *board*, built once and cached.

    Safe under id(board) reuse (see :data:`_CHARUCO_DETECTOR_CACHE`'s
    docstring): a cache hit is only trusted when the entry's own board
    ``is`` *board*, so a collision with a garbage-collected board's old
    address always rebuilds instead of returning a stale, wrong-shaped
    detector. Because an entry holds a strong reference to its board, the
    cache is bounded (:data:`_CHARUCO_DETECTOR_CACHE_MAXSIZE`, least-
    recently-used eviction) rather than left to grow forever; evicting an
    entry is also what lets its board be collected and its id() safely
    recycled, since eviction removes the dict key a later collision would
    otherwise be checked against.

    Mutating *board* in place (e.g. ``board.setLegacyPattern(...)``) still
    reaches a cached detector built from it: the cache holds the same board
    object, not a copy, so this is unaffected by the identity check above.

    Built through the same :meth:`ArucoOpenCVDetector.build_detector` helper
    the aruco1 path uses, for configuration parity with it -- though this is
    belt-and-braces rather than load-bearing: empirically indistinguishable
    from a bare ``cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters())``
    on this path, because the ``DetectorParameters``/``RefineParameters`` the
    helper also sets only change anything when OpenCV runs its *own*
    (re)detection against rejected marker candidates -- which never happens
    here, since markers are always supplied directly (see
    :func:`interpolate_board_corners`).

    The whole lookup/build/evict sequence runs under
    :data:`_CHARUCO_DETECTOR_CACHE_LOCK` (round-2 review, P2): a bare dict's
    check-then-delete eviction is not atomic and can raise under genuine
    multi-threaded access (see the lock's own docstring).
    """
    key = id(board)
    with _CHARUCO_DETECTOR_CACHE_LOCK:
        entry = _CHARUCO_DETECTOR_CACHE.get(key)
        if entry is not None and entry[0] is board:
            # Re-insert at the end so eviction below stays least-recently-used
            # (plain dicts are insertion-ordered; this is that, not a hash quirk).
            del _CHARUCO_DETECTOR_CACHE[key]
            _CHARUCO_DETECTOR_CACHE[key] = entry
            return entry[1]

        detector = ARUCO_OPENCV_DETECTOR.build_detector(
            board, ARUCO_OPENCV_DETECTOR.resolve(None))
        _CHARUCO_DETECTOR_CACHE[key] = (board, detector)
        if len(_CHARUCO_DETECTOR_CACHE) > _CHARUCO_DETECTOR_CACHE_MAXSIZE:
            oldest_key = next(iter(_CHARUCO_DETECTOR_CACHE))
            del _CHARUCO_DETECTOR_CACHE[oldest_key]
        return detector


#: Revision of the marker quads :func:`interpolate_board_corners` hands to
#: ``detectBoard``. Part of a detection cache's identity for a classic
#: (ChArUco, Ccube, CIco) target read with aruco2, so a cache written by an
#: earlier revision is a miss rather than reused. Bump it whenever the
#: quads, and so the corners, a given image yields can change.
CHARUCO_SEED_REVISION = 1

#: Slack, in pixels, added to aruco2's sub-pixel half-window when deciding
#: whether that window could have reached the neighbouring chessboard
#: corner (see :func:`_saddle_reachable`).
_SEED_GATE_SLACK_PX = 1.5

#: A replacement quad whose corners move further than this fraction of the
#: aruco2 quad's side is not the same marker reading, and is not used.
_SEED_MAX_SHIFT_FRACTION = 0.3


def _gray_of(image: np.ndarray) -> np.ndarray:
    """A single-channel view of a uint8 image, converted as OpenCV's own
    marker detection converts a colour image."""
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _mean_side(quad: np.ndarray) -> float:
    """Mean side length of a (4, 2) quad, in pixels."""
    return float(np.mean(np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)))


def _aruco2_subpix_halfwin(sides: list[float]) -> int:
    """The cornerSubPix half-window aruco2 refined these markers with.

    aruco2 refines every marker corner in an image with ONE window, sized
    from the mean marker side L: ``min(int(3 * max(1, L / 34)), 9)``, so
    never below 3 px (aruco2_fork src/aruco2.cpp, the corner-refinement
    block after marker identification). Here L is the mean over the markers
    this call was given, which for a cube face is that face's markers: the
    caller's marker list is all this function sees.
    """
    return min(int(3 * max(1.0, float(np.mean(sides)) / 34.0)), 9)


def _saddle_reachable(side: float, marker_fraction: float, halfwin: int) -> bool:
    """Whether aruco2's window could have pulled this marker's corners onto
    the neighbouring chessboard corner.

    On a classic ChArUco a marker corner sits ``(1 - f) / (2 f) * side``
    pixels from the adjacent chessboard corner along each axis (f = marker
    length / square length). When that is inside the half-window (plus a
    little slack), cornerSubPix sees the stronger chessboard saddle and
    drags the marker corner towards it.
    """
    gap = (1.0 - marker_fraction) / (2.0 * marker_fraction) * side
    return gap < halfwin + _SEED_GATE_SLACK_PX


def _opencv_quads(gray: np.ndarray, dictionary, quads: list[np.ndarray]) -> dict[int, np.ndarray]:
    """OpenCV's own, unrefined marker quads near the given quads, by id.

    Detection runs on a crop around *quads* rather than the whole image: a
    cube face's markers occupy a small part of it, and the call is repeated
    per face. It reads as the whole image would: the margin keeps OpenCV's
    adaptive threshold (window at most 23 px) clear of the crop edge, and
    the marker perimeter limits, which OpenCV states relative to the image's
    larger side, are rescaled so they stay the same in pixels.
    """
    pts = np.concatenate(quads, axis=0)
    margin = max(32.0, 2.0 * max(_mean_side(q) for q in quads))
    height, width = gray.shape[:2]
    x0 = int(max(0, np.floor(pts[:, 0].min() - margin)))
    y0 = int(max(0, np.floor(pts[:, 1].min() - margin)))
    x1 = int(min(width, np.ceil(pts[:, 0].max() + margin) + 1))
    y1 = int(min(height, np.ceil(pts[:, 1].max() + margin) + 1))
    if x1 <= x0 or y1 <= y0:
        return {}
    params = cv2.aruco.DetectorParameters()
    scale = max(height, width) / max(y1 - y0, x1 - x0)
    params.minMarkerPerimeterRate = params.minMarkerPerimeterRate * scale
    params.maxMarkerPerimeterRate = params.maxMarkerPerimeterRate * scale
    # Forced, never taken from a caller's settings: the point is OpenCV's
    # contour corners. SUBPIX would bring back a window bias, and APRILTAG
    # returns corners in AprilTag's pixel convention, about +0.45 px in x
    # and y from OpenCV's (OpenCV 5.0.0 aruco_detector.cpp, delta 0.5).
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _rejected = detector.detectMarkers(
        np.ascontiguousarray(gray[y0:y1, x0:x1]))
    if ids is None:
        return {}
    offset = np.array([x0, y0], dtype=np.float32)
    found: dict[int, np.ndarray] = {}
    seen: set[int] = set()
    for c, i in zip(corners, np.asarray(ids).reshape(-1)):
        i = int(i)
        if i in seen:
            # Read twice: ambiguous, so neither copy is used.
            found.pop(i, None)
            continue
        seen.add(i)
        found[i] = np.asarray(c, dtype=np.float32).reshape(4, 2) + offset
    return found


def _edge_refit(gray: np.ndarray, quad: np.ndarray, marker_fraction: float,
                origin: np.ndarray) -> np.ndarray:
    """Re-estimate a marker quad from its own outer edges.

    Each side's position is measured along the middle 60 % of the side,
    away from the corners (which is where aruco2's window drags them), and
    the four fitted lines are intersected. *gray* is a lightly blurred
    float32 crop of the image whose top-left pixel is *origin* (integer
    x, y) in the frame of *quad*.

    Along each side's outward normal the search runs from
    ``-max(2, 0.2 * side)`` to ``+max(0.6, 0.5 * gap)`` pixels, gap being the
    white border between the marker and the neighbouring black square, and
    takes the strongest dark-to-bright step. A side with fewer than three
    usable samples, near-parallel neighbouring sides, or a result that moves
    any corner by more than ``_SEED_MAX_SHIFT_FRACTION * side`` keeps the
    quad as it was.
    """
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    side = _mean_side(q)
    centre = q.mean(axis=0)
    gap = (1.0 - marker_fraction) / (2.0 * marker_fraction) * side
    offsets = np.arange(-max(2.0, 0.2 * side), max(0.6, 0.5 * gap) + 1e-9, 0.25)
    lines = []
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        direction = (b - a) / np.linalg.norm(b - a)
        normal = np.array([direction[1], -direction[0]])
        if np.dot(normal, (a + b) / 2 - centre) < 0:
            normal = -normal  # point outwards, away from the marker
        n_samples = max(5, int(np.linalg.norm(b - a) / 1.5))
        ts = np.linspace(0.2, 0.8, n_samples)
        base = a[None] + ts[:, None] * (b - a)[None]
        # One intensity profile per sample point, across the edge.
        grid = base[:, None, :] + offsets[None, :, None] * normal[None, None, :]
        # Rounded to float32 in the image's own frame, then moved into the
        # crop's: subtracting an integer origin is exact, so remap samples
        # exactly where it would on the whole image.
        grid = grid.astype(np.float32) - origin.astype(np.float32)
        profile = cv2.remap(gray, np.ascontiguousarray(grid[..., 0]), np.ascontiguousarray(grid[..., 1]),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        step = np.diff(profile, axis=1)  # dark marker -> white border is positive
        best = np.argmax(step, axis=1)
        edge_points = []
        for r in range(len(ts)):
            j = best[r]
            if step[r, j] <= 3.0:  # no real edge on this profile
                continue
            frac = 0.0
            if 0 < j < step.shape[1] - 1:
                # Parabolic peak interpolation between neighbouring steps.
                y0, y1, y2 = step[r, j - 1], step[r, j], step[r, j + 1]
                den = y0 - 2 * y1 + y2
                if den < 0:
                    frac = 0.5 * (y0 - y2) / den
            # A difference sits half-way between its two samples (0.25 px apart).
            offset = offsets[j] + 0.125 + frac * 0.25
            edge_points.append(base[r] + offset * normal)
        if len(edge_points) < 3:
            return quad
        edge_points = np.asarray(edge_points)
        mean = edge_points.mean(axis=0)
        # Total-least-squares line through the edge points.
        _, _, vt = np.linalg.svd(edge_points - mean)
        lines.append((mean, vt[0]))
    refit = []
    for k in range(4):
        # Corner k is where side k-1 meets side k.
        (c1, d1), (c0, d0) = lines[k], lines[(k - 1) % 4]
        system = np.column_stack([d0, -d1])
        if abs(np.linalg.det(system)) < 1e-6:
            return quad
        s = np.linalg.solve(system, c1 - c0)
        refit.append(c0 + s[0] * d0)
    refit = np.asarray(refit)
    if np.max(np.linalg.norm(refit - q, axis=1)) > _SEED_MAX_SHIFT_FRACTION * side:
        return quad
    return refit.astype(np.float32)


def _refit_quads(gray: np.ndarray, quads: list[np.ndarray], marker_fraction: float) -> list[np.ndarray]:
    """:func:`_edge_refit` on each quad, blurring only the region they need.

    The crop's margin covers the edge search and the blur kernel, so the
    result is the one a whole-image blur would give.
    """
    pts = np.concatenate(quads, axis=0)
    margin = max(_mean_side(q) for q in quads) + 10.0
    height, width = gray.shape[:2]
    x0 = int(max(0, np.floor(pts[:, 0].min() - margin)))
    y0 = int(max(0, np.floor(pts[:, 1].min() - margin)))
    x1 = int(min(width, np.ceil(pts[:, 0].max() + margin) + 1))
    y1 = int(min(height, np.ceil(pts[:, 1].max() + margin) + 1))
    if x1 <= x0 or y1 <= y0:
        return list(quads)
    blurred = cv2.GaussianBlur(gray[y0:y1, x0:x1].astype(np.float32), (0, 0), 0.6)
    origin = np.array([x0, y0])
    return [_edge_refit(blurred, q, marker_fraction, origin) for q in quads]


def _seed_quads(image: np.ndarray, board, markers: list) -> list:
    """aruco2's marker quads, corrected where aruco2's own refinement
    biased them, ready for ``detectBoard``.

    Why: aruco2 refines every marker corner with one image-wide cornerSubPix
    window of at least 3 px (see :func:`_aruco2_subpix_halfwin`). On a
    classic ChArUco with small squares that window reaches the adjacent
    chessboard corner and drags the marker corner towards it, and
    ``detectBoard``'s own corner refinement (a 1 px half-window at those
    sizes) cannot undo the resulting error in the interpolated seed, so the
    ChArUco corner comes back biased by a pixel or more.

    What, per marker, ids and order unchanged:

    1. where OpenCV's unrefined ``detectMarkers`` (the board's dictionary,
       read near these markers) reads the same id, once, with every corner
       within ``_SEED_MAX_SHIFT_FRACTION`` of a side of aruco2's, OpenCV's
       quad is used;
    2. otherwise, where aruco2's window could have reached the chessboard
       corner (:func:`_saddle_reachable`), the quad is refitted from its
       own edges (:func:`_edge_refit`);
    3. otherwise aruco2's quad is kept.

    Deterministic: no randomness, and the result depends only on the image,
    the board and the marker list.
    """
    gray = _gray_of(image)
    quads = [corners for _mid, corners in markers]
    sides = [_mean_side(q) for q in quads]
    opencv = _opencv_quads(gray, board.getDictionary(), quads)

    seeded = list(quads)
    to_refit = []
    halfwin = _aruco2_subpix_halfwin(sides)
    marker_fraction = float(board.getMarkerLength()) / float(board.getSquareLength())
    for k, (mid, quad) in enumerate(markers):
        other = opencv.get(mid)
        if other is not None and (np.max(np.linalg.norm(other - quad, axis=1))
                                  <= _SEED_MAX_SHIFT_FRACTION * sides[k]):
            seeded[k] = other
        elif _saddle_reachable(sides[k], marker_fraction, halfwin):
            to_refit.append(k)
    if to_refit:
        refitted = _refit_quads(gray, [quads[k] for k in to_refit], marker_fraction)
        for k, quad in zip(to_refit, refitted):
            seeded[k] = quad
    return [(mid, np.ascontiguousarray(quad, dtype=np.float32))
            for (mid, _corners), quad in zip(markers, seeded)]


def detect_charuco_corners(image, board, dict_int, warn_legacy=None):
    """D4/D5: detect markers with aruco2 and interpolate ChArUco corners.

    :param image: the (uint8) image to detect in.
    :param board: a cv2.aruco.CharucoBoard defining the target geometry.
    :param dict_int: the dictionary int used for detection.
    :param warn_legacy: optional zero-arg callable; see
        :func:`interpolate_board_corners`.
    :return: (corner_ids 1D int array, pts (N,2) float32) or (None, None).
    """
    image = _as_uint8_image(image)
    markers = detect_markers(image, dict_int)
    return interpolate_board_corners(image, board, markers, warn_legacy=warn_legacy)


def interpolate_board_corners(image, board, markers, warn_legacy=None):
    """D4: interpolate ChArUco corners from aruco2-detected markers.

    An adapter, not a reimplementation: aruco2's marker quads, corrected
    where aruco2's own corner refinement is known to bias them on small
    squares (:func:`_seed_quads`; ids are never changed), are handed to
    OpenCV's native ``CharucoDetector.detectBoard``
    (via :func:`_charuco_detector_for`), which does the per-marker
    homography, corner matching and sub-pixel refinement itself -- the
    same code the aruco1 path already runs on its own, natively-detected
    markers. aruco2's marker detection (:func:`detect_markers`) and
    Ccube's global-id-to-face-local-id remap (done by the caller before
    this function ever sees the markers) are unchanged.

    A single visible marker never yields corners: ``CharucoParameters.
    minMarkers`` is left at OpenCV's default of 2, deliberately, rather than
    lowered to 1. This is parity with the aruco1 path, which reads the same
    default; a lone marker still fixes a corner's position by extrapolation
    (one marker's own homography, not a triangulation between several), and
    that used to be a source of this path's grosser errors before the
    switch to this CharucoDetector-based adapter. A frame with only one
    marker in view is therefore skipped rather than answered inaccurately.

    Fail-safe by construction, not by a corrective heuristic: beyond
    dropping a duplicate marker id (an ambiguous INPUT -- which of two
    same-id markers is real cannot be told here, never a signal that either
    one is corrupt), a frame OpenCV's own ``detectBoard`` rejects outright
    is returned as no corners, full stop. An earlier revision of this
    module also tried to recover such a frame by dropping whichever single
    marker looked geometrically inconsistent with the rest -- a round-1
    review found that heuristic could itself return WRONG, mislabelled
    corners: under a legacy-pattern mismatch on a partial view, a board's
    real, uncorrupted markers can split into a majority/minority group
    that each look internally consistent, and pruning the minority as "the
    bad one" let detection succeed anyway, under the still-wrong legacy
    flag, with no warning. That heuristic was removed rather than patched,
    because a corner interpolator that can turn a safe empty result into a
    confidently wrong one violates this module's whole reason for being
    conservative here (fewer corners is acceptable, wrong corners are not
    -- see the module docstring).

    Approved policy (legacy pattern): detection reads ONLY the legacy flag
    *board* is configured with when this function is called -- no per-frame
    retry under the opposite one, and *board*'s own legacy flag is never
    changed here. A frame that finds markers but no corners under that
    configured flag may still trigger a WARNING (see *warn_legacy* below)
    when a throwaway probe under the opposite pattern gives strong evidence
    of a mismatch, but the probe's own corners are evidence only and are
    never returned. Accepted trade-off, not a bug: on a partial view of a
    misconfigured board, a small marker subset can occasionally be
    self-consistent enough to interpolate SOME corners even under the
    wrong, configured pattern; the warning is the mitigation for that, not
    a guarantee it cannot happen. See ``markers/legacy_probe.py``.

    :param image: the (uint8) image the markers were detected in.
    :param board: a cv2.aruco.CharucoBoard whose geometry defines the corners.
    :param markers: list of (marker_id, corners (4,2) float32) with ids in the
        board's local id space (Ccube callers remap global ids first).
    :param warn_legacy: optional zero-arg callable fired at most once
        (per-target de-duplication is the caller's own responsibility) when
        markers were found but no ChArUco corners could be interpolated
        under *board*'s configured legacy-pattern flag, AND a throwaway
        probe under the opposite pattern gives strong evidence of a
        mismatch -- see ``markers.legacy_probe.should_warn_legacy_mismatch``
        for exactly what that requires. Mirrors the aruco1 path's own
        warning point (``charuco/target.py``'s ``find_in_image``). Pass
        ``None`` once the caller has already warned: that also skips the
        probe itself, not just the (no-op) callback, so a caller should
        stop passing a real callable once its own de-duplication state
        says the warning has already fired.
    :return: (corner_ids 1D int32 array, pts (N,2) float32) or (None, None).
    """
    image = _as_uint8_image(image)

    # aruco2's detect_markers always returns (4,2) float32 quads, so this
    # guards the public seam against a malformed or foreign marker list
    # rather than validating real input. detectBoard itself tolerates
    # out-of-bounds and foreign ids; an id outside int32 range is
    # dropped here because the cast below would raise on it.
    good = []
    for mid, corners in markers:
        corners = np.asarray(corners, dtype=np.float32)
        if corners.shape != (4, 2):
            _LOG.debug(
                "aruco2: skipping marker id %s with shape %s (need (4, 2))",
                mid, corners.shape)
            continue
        mid = int(mid)
        if not (_INT32_MIN <= mid <= _INT32_MAX):
            _LOG.debug(
                "aruco2: skipping marker id %s outside int32 range %s..%s "
                "(detectBoard's own marker-id cast would overflow)",
                mid, _INT32_MIN, _INT32_MAX)
            continue
        good.append((mid, corners))
    if not good:
        # An EMPTY markerCorners container makes detectBoard silently run
        # its own aruco1-style (re)detection instead of skipping straight
        # to "nothing to interpolate" -- short-circuit before that happens.
        return None, None

    # An id seen twice is ambiguous, and detectBoard rejects the whole
    # frame over it rather than choosing. Drop every marker sharing a
    # repeated id, not just the extras.
    id_counts: dict[int, int] = {}
    for mid, _corners in good:
        id_counts[mid] = id_counts.get(mid, 0) + 1
    duplicate_ids = {mid for mid, count in id_counts.items() if count > 1}
    if duplicate_ids:
        _LOG.debug("aruco2: dropping duplicate-id marker(s) %s", sorted(duplicate_ids))
        good = [(mid, corners) for mid, corners in good if mid not in duplicate_ids]
    if not good:
        return None, None

    # aruco2's quads are biased towards the chessboard corners on small
    # squares; detectBoard is seeded with corrected ones (see _seed_quads).
    good = _seed_quads(image, board, good)

    detector = _charuco_detector_for(board)

    def _detect(marker_list):
        marker_corners = tuple(corners.reshape(1, 4, 2) for _mid, corners in marker_list)
        # int32, not the platform default int: float64/int64 ids raise
        # inside OpenCV's own binding for this call.
        marker_ids = np.asarray(
            [mid for mid, _corners in marker_list], dtype=np.int32).reshape(-1, 1)
        return detector.detectBoard(image, markerCorners=marker_corners, markerIds=marker_ids)

    c_corners, c_ids, mloc, mid = _detect(good)

    if c_corners is None and mloc is not None:
        # No retry: the configured pattern's "no corners" stands, and the
        # probe only decides whether to warn. None once the caller has
        # warned, which also skips the probe's cost from then on.
        if warn_legacy is not None and should_warn_legacy_mismatch(
            board, ARUCO_OPENCV_DETECTOR.resolve(None), image,
            len(mloc), mloc, mid,
        ):
            warn_legacy()

    if c_corners is None:
        return None, None

    ids = np.asarray(c_ids).reshape(-1).astype(np.int32)
    pts = np.asarray(c_corners).reshape(-1, 2).astype(np.float32)
    return ids, pts


class Aruco2Detector(DetectorParameterisation):
    """
    The aruco2 package's marker detector, which takes no settings.

    ``aruco2.detect_fiducial_markers`` is given an image and a dictionary and
    nothing else, so there is nothing here for a form to show or a study to
    sweep.  Said as an empty parameterisation rather than left undescribed:
    a target reading its markers this way used to accept OpenCV's settings,
    warn that they were being ignored, and ignore them.
    """

    name = ARUCO2_BACKEND

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return ()

    def unavailable_reason(self, values: dict | None = None) -> str | None:
        if marker_backend_available(ARUCO2_BACKEND):
            return None
        return (
            "ArUco 2 (aruco2) is selected but the 'aruco2' package is not "
            "installed. It is not on PyPI: build it from the third_party/aruco2 "
            "submodule (see CITATION.md), or switch the marker backend to "
            "ArUco 1 (OpenCV).")


#: Shared rather than built per target: it describes nothing per instance.
ARUCO2_DETECTOR = Aruco2Detector()
