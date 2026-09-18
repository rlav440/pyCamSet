'''
:Purpose: ArUco2 (aruco2 package) marker backend for pyCamSet ChArUco/Ccube
    targets. Implements plan v4 decisions D3 (dictionary resolution) and
    D5 (marker detection) for the aruco2 backend; the aruco1 (OpenCV) path
    is untouched. ChArUco corner interpolation (D4) is an adapter: aruco2's
    own markers are handed straight to a cached ``cv2.aruco.CharucoDetector``
    per board, so the same native interpolation and sub-pixel refinement the
    aruco1 path uses also reads aruco2's markers -- there is no separate
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
# The corner-interpolation adapter (below) hands aruco2's markers to
# OpenCV's own CharucoDetector, built through the SAME helper the aruco1
# path uses -- for configuration parity, not because it changes anything
# observable here (see _charuco_detector_for's docstring).
from pyCamSet.calibration_targets.markers.aruco_opencv import ARUCO_OPENCV_DETECTOR
from pyCamSet.calibration_targets.markers.legacy_probe import should_warn_legacy_mismatch

# Module-level lazy aruco2 import guard (D3): the module must import cleanly
# even when aruco2 is not installed. ARUCO2_AVAILABLE is the single source of
# truth for availability checks in the target classes.
#
# ImportError means the package is not on the path at all (not installed).
# OSError means the package *was found* -- its .pyd loaded far enough for the
# loader to try to import it -- but pulling in its native dependencies (the
# Microsoft Visual C++ runtime DLLs the aruco2 wheel links against, such as
# CONCRT140.dll, MSVCP140.dll or VCRUNTIME140.dll) failed. Those are two
# different problems with two different fixes, so the OSError text is kept
# for _require_aruco2() to report rather than folded into the same "not
# installed" message.
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


#: Bounds a marker id must fall within to survive _detect()'s own
#: ``np.asarray(..., dtype=np.int32)`` cast further down this module: an id
#: outside this range raises OverflowError from numpy rather than being
#: skipped the way an out-of-bounds/foreign id normally is (round-3
#: finding, P3). See the ``good = []`` filtering loop in
#: :func:`interpolate_board_corners`.
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


#: Bounds how many (board, detector) entries _CHARUCO_DETECTOR_CACHE keeps
#: alive at once. Arbitrary but generous: a target holds at most a handful
#: of live boards (a ChArUco one, six for a Ccube), so this is not expected
#: to bind in normal use -- it exists so the cache cannot grow without
#: bound across many short-lived boards (e.g. a study that rebuilds targets
#: repeatedly), which the strong board reference below would otherwise do.
_CHARUCO_DETECTOR_CACHE_MAXSIZE = 64

#: One cv2.aruco.CharucoDetector per board, built once and reused. Keyed by
#: id(board), but each entry ALSO holds a strong reference to the board
#: itself: CPython reuses a garbage-collected object's address for a new
#: object (verified empirically: alternating two differently-shaped boards
#: through a construct/delete/gc loop reused ids for most iterations), so a
#: bare id()-keyed cache can hand a new, differently-shaped board the OLD
#: board's detector -- silently, since detectBoard never re-validates that
#: its detector's own board matches the geometry the caller thinks it is
#: reading. Keeping the board alongside its detector lets a lookup check
#: identity (``is``) before reusing an entry, which is correct regardless
#: of whether an id() match is a genuine hit or a collision: a collision
#: simply misses (rebuilds), because the stored board is never the one
#: being asked for. See :func:`_charuco_detector_for`.
_CHARUCO_DETECTOR_CACHE: dict[int, tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]] = {}

#: Guards every read-check-insert-evict sequence on
#: :data:`_CHARUCO_DETECTOR_CACHE` (round-2 review, P2). The cache is
#: process-local -- pyCamSet's own ``multiprocessing.Pool`` workers each get
#: an independent copy, so this is not about that -- but a caller that drives
#: this backend from real OS threads (a GUI worker thread, a
#: ``ThreadPoolExecutor`` pipeline) could otherwise race the dict's
#: check-then-delete eviction step: two threads evicting concurrently can
#: compute the same "oldest" key and both try to delete it (``KeyError``), or
#: a mutation can land between another thread's ``iter()`` and its lookup
#: (``RuntimeError: dictionary changed size during iteration``). Both were
#: reproduced directly against this cache under a widened thread-switch
#: interval. The common path is cache-hit-dominated, so a single lock around
#: the whole lookup/build/evict body costs it a negligible uncontended
#: acquisition.
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

    An adapter, not a reimplementation: aruco2's own marker corners are
    handed straight to OpenCV's native ``CharucoDetector.detectBoard``
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

    # aruco2's own detect_markers() always returns (4,2) float32 quads, so
    # this is not a validation pass over well-formed real input -- it is
    # what keeps a malformed/foreign marker list (the regression test
    # below; interpolate_board_corners is a public seam) from being cast
    # into one inhomogeneous array. OpenCV's own detectBoard needs no
    # finite/area/bounds guarding beyond that: verified to tolerate
    # out-of-bounds (cropped/partial-view) and foreign-id markers without
    # raising, unlike the D4-step homography this replaces -- EXCEPT an id
    # outside int32 range, which _detect()'s own np.int32 cast (below) would
    # otherwise raise OverflowError on (round-3 finding, P3): drop it here,
    # the same fail-safe way a wrong-shaped quad is dropped, rather than let
    # a single foreign id abort the whole frame.
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

    # An id seen twice in one frame is ambiguous -- which of the two is the
    # real marker cannot be told here -- and detectBoard's own consistency
    # check rejects the WHOLE frame over it rather than picking one. Drop
    # every marker sharing a repeated id (not just the extras) before it
    # ever reaches detectBoard.
    id_counts: dict[int, int] = {}
    for mid, _corners in good:
        id_counts[mid] = id_counts.get(mid, 0) + 1
    duplicate_ids = {mid for mid, count in id_counts.items() if count > 1}
    if duplicate_ids:
        _LOG.debug("aruco2: dropping duplicate-id marker(s) %s", sorted(duplicate_ids))
        good = [(mid, corners) for mid, corners in good if mid not in duplicate_ids]
    if not good:
        return None, None

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
        # Approved policy: no retry -- board's own legacy flag is never
        # touched here, and the configured pattern's own "no corners" result
        # stands. The only thing left to decide is whether this frame is
        # strong enough evidence of a mismatch to warn about (see
        # markers.legacy_probe.should_warn_legacy_mismatch's docstring for
        # exactly what "strong" means); a throwaway probe board/detector is
        # used for that check, built for the OPPOSITE pattern, never
        # board's own -- its corners are evidence only, never returned.
        # warn_legacy is None once the caller has already warned (P2,
        # round-1 review): that also skips should_warn_legacy_mismatch's
        # own probe below via this short-circuit, so it is paid at most
        # once per target, not on every later qualifying frame.
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
