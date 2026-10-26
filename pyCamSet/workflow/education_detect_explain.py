"""Explain, step by step and as drawable data, how Phase 1 found the target in one photo.

sees", "Finding markers"). The recorded Phase 1 trace keeps only IDs per step, so this module
*recomputes* the explanation live for one camera + image.

How it stays honest:

* The detection itself is Phase 1's own per-image path,
  :func:`~pyCamSet.calibration_targets.core.abstract_target.detect_in_image`, run with an
  ``explain`` collector. The detectors append what each of their own steps saw (default-off
  hooks: ``None`` in every normal run changes nothing). Every step built from those records has
  ``source == "detector"``.
* Some steps happen inside compiled code that reports nothing back (OpenCV's ArUco thresholding
  and candidate search, aruco2's marker finder, the PuzzleBoard corner filter). Those steps are
  *reproduced* from the same library calls with the detector's own parameters, for display, and
  are marked ``source == "reproduced"``; where the real run reports a result (accepted markers,
  OpenCV's rejected candidates) the reproduction is labelled with it and its agreement counted.
* :class:`ConsistencyCheck` compares the live result with the saved Phase 1 result.

All geometry is in the *detection frame*: the pixel frame of the saved Phase 1 detections (the
photo's own pixels, times ``upscale_factor`` when Phase 1 upscaled). Each step's ``image`` may
be smaller (downsampled for display, or the rescaled detector input); ``image_scale`` says how
many image pixels one detection-frame pixel is, so ``xy * image_scale`` lands on the image.
Points are in the pixel-centre frame every detector returns
(:data:`~pyCamSet.calibration_targets.markers.aruco_opencv.PIXEL_CONVENTION`), and a point
read on the rescaled detector input is mapped to the detection frame by the same pixel-centre
rule Phase 1 maps its detections with (:func:`_to_frame`).

Qt and matplotlib.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from pyCamSet.calibration_targets.core.abstract_target import (
    COORDINATE_MAPPING,
    _looks_like_native_opencv_error,
    detect_in_image,
    enumerate_detection_images,
    legacy_mapping_shift,
    read_detection_image,
)
from pyCamSet.calibration_targets.core.target_detections import ImageDetection
from pyCamSet.calibration_targets.core.preparation import (
    MIN_MAX,
    PREPARATION_DEFAULTS,
    PUZZLEBOARD,
    UPPER_BYTE,
    resolve_preparation,
    target_family,
)
from pyCamSet.calibration_targets.markers.aruco_opencv import (
    CHARUCO_DETECTOR_TARGETS,
    OPENCV_MAJOR,
    OPENCV_VERSION,
    PIXEL_CONVENTION,
)
from pyCamSet.calibration_targets.markers.puzzleboard import min_max_to_uint8

#: Longest side of a step image; geometry stays in the detection frame regardless.
#: 1024 px is what the stage can show at its largest size; 1600 px held about 2.4x the memory.
MAX_DISPLAY_SIDE = 1024
#: At most this many warped marker crops are kept for the gallery (accepted first).
MAX_CROPS = 160
#: At most this many contour outlines are kept for drawing.
MAX_CONTOURS = 4000
#: Side of a warped marker crop, in pixels.
CROP_SIDE = 64
#: How many sub-pixel insets a refinement step carries.
MAX_INSETS = 6


def _to_frame(points, frame_to_prepared: float) -> np.ndarray:
    """Points on the detector's (possibly rescaled) input image, in the detection frame.

    Both are pixel-centre, so this is the rule Phase 1 maps its detections back with
    (:func:`~pyCamSet.calibration_targets.core.abstract_target._restore_native_coordinates`):
    ``(u + 0.5) / scale - 0.5``. Dividing by the scale alone would put an overlay
    ``0.5 / scale - 0.5`` pixels off the points it explains.
    """
    points = np.asarray(points, dtype=np.float64)
    if frame_to_prepared == 1.0:
        return points
    return (points + 0.5) / frame_to_prepared - 0.5


def _to_prepared(points, frame_to_prepared: float) -> np.ndarray:
    """The inverse of :func:`_to_frame`: detection-frame points on the detector's input image."""
    points = np.asarray(points, dtype=np.float64)
    if frame_to_prepared == 1.0:
        return points
    return (points + 0.5) * frame_to_prepared - 0.5

#: Phase 1's defaults for the rescale-and-gamma preparation of a PuzzleBoard target.  Other
#: targets have their own defaults (:data:`~pyCamSet.calibration_targets.core.preparation.
#: PREPARATION_DEFAULTS`); :func:`explain_detection` looks them up by target.
DEFAULT_PREPROCESSING_SCALE = PREPARATION_DEFAULTS[PUZZLEBOARD].scale
DEFAULT_PREPROCESSING_GAMMA = PREPARATION_DEFAULTS[PUZZLEBOARD].gamma


# ---------------------------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------------------------

@dataclass
class Overlay:
    """Typed geometry to draw on a step image, in detection-frame pixels.

    ``kind`` is one of:

    * ``"points"`` -- ``data`` is ``(N, 2)``;
    * ``"polygons"`` / ``"polylines"`` -- ``data`` is a list of ``(K, 2)`` arrays (closed / open);
    * ``"segments"`` -- ``data`` is ``(N, 2, 2)``, from ``[:, 0]`` to ``[:, 1]`` (e.g. a move);
    * ``"labels"`` -- ``data`` is ``(N, 2)`` anchor points for ``labels``.

    ``states`` (one per item, optional) is a small vocabulary the GUI maps to colours:
    ``"kept"``, ``"rejected"``, ``"accepted"``, ``"removed"``, ``"moved"``, ``"unchanged"``,
    ``"predicted"``, ``"candidate"``, ``"dropped"``, ``"bit_ok"``, ``"bit_wrong"``.
    ``reasons`` (one per item, optional) says in plain words why an item was rejected.
    """

    kind: str
    name: str
    data: Any
    states: Optional[list[str]] = None
    labels: Optional[list[str]] = None
    reasons: Optional[list[str]] = None

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class Inset:
    """A magnified crop around one point, e.g. a corner before and after sub-pixel refinement.

    ``image`` is the crop (uint8, gray or RGB) magnified ``zoom`` times; ``origin`` is the
    detection-frame position of its top-left pixel corner, so a detection-frame point ``p`` is
    drawn at ``(p - origin) * zoom * frame_to_crop`` (``frame_to_crop`` is the crop's own
    pixels per detection-frame pixel, 1 unless the detector worked on a rescaled image).
    """

    title: str
    image: np.ndarray
    origin: tuple[float, float]
    zoom: float
    frame_to_crop: float
    before: Optional[np.ndarray] = None
    after: Optional[np.ndarray] = None
    label: str = ""
    state: str = ""


@dataclass
class MarkerCrop:
    """One square candidate, warped flat, with the bit grid read from it.

    ``bits`` is the ``(n+2, n+2)`` grid including the one-cell border, 1 = black, or None when
    no grid was read. ``state`` is ``"accepted"`` or the reason it failed:
    ``"bad_border"``, ``"not_in_dictionary"``, ``"too_uniform"``. ``marker_id``/``rotation``
    are what the reading produced (``-1`` if nothing); ``detector_id`` is the ID the real run
    accepted at this place (or ``None``), so a disagreement is visible.
    """

    image: np.ndarray
    bits: Optional[np.ndarray]
    marker_bits: int
    state: str
    marker_id: int
    rotation: int
    polygon: np.ndarray
    detector_id: Optional[str] = None
    face: Optional[int] = None
    source: str = "reproduced"
    #: True for a white-bordered marker, read from the inverted image (aruco2 grid boards).
    inverted: bool = False
    #: For a read marker (display only): the inner code as the dictionary stores it
    #: (``n x n``, 1 = white cell of a black-bordered marker), turned to the dictionary's
    #: orientation; ``turns`` quarter turns were needed.
    code: Optional[np.ndarray] = None
    turns: int = 0
    #: The dictionary entry it matched and two neighbouring entries: ``[(id, bits), ...]``
    #: (same convention as ``code``), for the "code -> ID" lookup picture.
    lookup: list = field(default_factory=list)
    #: Cells of the read code that differ from the dictionary entry (fixed by error correction).
    bit_errors: int = 0
    #: Number of markers in the dictionary that was searched.
    dictionary_size: int = 0
    #: Display only (SCI3-14 / ED3-10): cells by which the closest *other* code differs, how
    #: many other codes share that distance, and how many misread cells the reading corrects.
    closest_distance: int = -1
    closest_ties: int = 0
    correctable_cells: int = 0


@dataclass
class ExplainStep:
    """One processing step: one picture, one caption line, a few stat chips."""

    key: str
    title: str
    caption: str
    image: Optional[np.ndarray]
    image_scale: float = 1.0
    overlays: list[Overlay] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)
    crops: list[MarkerCrop] = field(default_factory=list)
    insets: list[Inset] = field(default_factory=list)
    images: list[tuple[str, np.ndarray]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    source: str = "detector"


@dataclass
class ConsistencyCheck:
    """Does the live recompute match the saved Phase 1 detection for this photo?"""

    available: bool
    ok: bool
    live_count: int
    saved_count: int
    matched: int = 0
    missing_keys: list = field(default_factory=list)
    extra_keys: list = field(default_factory=list)
    max_shift_px: float = float("nan")
    message: str = ""
    #: Keys the saved run holds more than once (a point decoded at two places). Detectors
    #: since 2026-10-08 drop such keys, so a run saved before that keeps them.
    saved_duplicate_keys: list = field(default_factory=list)


@dataclass
class DetectionExplanation:
    """Every step of one detection, in order, plus the result and its check."""

    steps: list[ExplainStep]
    backend: str
    target_type: str
    keys: np.ndarray
    image_points: np.ndarray
    consistency: ConsistencyCheck
    seconds: float
    photo_shape: tuple
    photo_dtype: str
    error: Optional[str] = None

    def step(self, key: str) -> Optional[ExplainStep]:
        for item in self.steps:
            if item.key == key:
                return item
        return None

    @property
    def step_keys(self) -> list[str]:
        return [item.key for item in self.steps]


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------

def _gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return (image >> 8).astype(np.uint8)
    if np.issubdtype(image.dtype, np.floating):
        top = float(np.nanmax(image)) if image.size else 1.0
        scale = 255.0 if top <= 1.0 else 1.0
        return np.clip(image * scale, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def _display(image: np.ndarray, frame_to_image: float, *, bgr: bool = True,
             max_side: int = MAX_DISPLAY_SIDE) -> tuple[np.ndarray, float]:
    """A uint8 display copy (RGB or gray), downsampled to *max_side*, and its frame scale."""
    shown = _to_uint8(image)
    if shown.ndim == 3:
        if shown.shape[2] == 4:
            shown = shown[..., :3]
        if bgr:
            shown = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
    longest = max(shown.shape[:2])
    factor = 1.0
    if longest > max_side:
        factor = max_side / float(longest)
        interp = cv2.INTER_NEAREST if shown.ndim == 2 and _is_binary(shown) else cv2.INTER_AREA
        shown = cv2.resize(shown, None, fx=factor, fy=factor, interpolation=interp)
    return np.ascontiguousarray(shown), float(frame_to_image * factor)


def _is_binary(image: np.ndarray) -> bool:
    sample = image[:: max(1, image.shape[0] // 64), :: max(1, image.shape[1] // 64)]
    return bool(np.all((sample == 0) | (sample == 255)))


def _is_grey_content(image: np.ndarray) -> bool:
    """True when a photo carries no colour (one channel, or all channels equal)."""
    image = np.asarray(image)
    if image.ndim == 2:
        return True
    sample = image[:: max(1, image.shape[0] // 128), :: max(1, image.shape[1] // 128), :3]
    return bool(np.all(sample == sample[..., :1]))


def _key_tuple(key) -> tuple:
    flat = np.asarray(key).reshape(-1)
    return tuple(int(v) for v in flat)


def _key_label(key) -> str:
    flat = _key_tuple(key)
    return ":".join(str(v) for v in flat)


def _polygon_centres(polygons: list[np.ndarray]) -> np.ndarray:
    if not polygons:
        return np.zeros((0, 2))
    return np.array([np.asarray(p, dtype=np.float64).reshape(-1, 2).mean(axis=0) for p in polygons])


def _match_polygons(a: list[np.ndarray], b: list[np.ndarray], tolerance: float) -> list[int]:
    """For each polygon of *a*, the index of the polygon of *b* with the nearest centre, or -1."""
    if not a:
        return []
    if not b:
        return [-1] * len(a)
    ca, cb = _polygon_centres(a), _polygon_centres(b)
    distance = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    nearest = distance.argmin(axis=1)
    match = [int(j) if distance[i, j] <= tolerance else -1 for i, j in enumerate(nearest)]
    # one-to-one: a marker claims only its closest candidate
    for j in set(m for m in match if m >= 0):
        claimants = [i for i, m in enumerate(match) if m == j]
        best = min(claimants, key=lambda i: distance[i, j])
        for i in claimants:
            if i != best:
                match[i] = -1
    return match


def _side_px(polygons: list[np.ndarray]) -> float:
    if not polygons:
        return 20.0
    sides = [np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).mean()
             for p in (np.asarray(q, dtype=np.float64).reshape(-1, 2) for q in polygons)]
    return float(np.median(sides))


def _inset(gray: np.ndarray, frame_to_gray: float, centre: np.ndarray, half_frame: float,
           before: np.ndarray, after: np.ndarray, title: str, label: str = "",
           state: str = "") -> Optional[Inset]:
    """A magnified crop of *gray* around *centre* (detection frame) with before/after points."""
    half = max(4, int(round(half_frame * frame_to_gray)))
    # Pixel-centre frame on both sides: gray pixel (i, j) is centred on (i, j).
    cx, cy = _to_prepared(np.asarray(centre, dtype=np.float64).reshape(2), frame_to_gray)
    x0, y0 = int(np.floor(cx + 0.5)) - half, int(np.floor(cy + 0.5)) - half
    h, w = gray.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + 2 * half + 1, w), min(y0 + 2 * half + 1, h)
    if xb - xa < 3 or yb - ya < 3:
        return None
    crop = gray[ya:yb, xa:xb]
    zoom = max(1.0, min(16.0, 192.0 / max(crop.shape[:2])))
    magnified = cv2.resize(_to_uint8(crop), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    # The crop's top-left pixel corner, (xa - 0.5, ya - 0.5) on gray, in the detection frame.
    return Inset(title=title, image=magnified,
                 origin=(xa / frame_to_gray - 0.5, ya / frame_to_gray - 0.5),
                 zoom=float(zoom), frame_to_crop=float(frame_to_gray),
                 before=np.asarray(before, dtype=np.float64).reshape(-1, 2),
                 after=np.asarray(after, dtype=np.float64).reshape(-1, 2),
                 label=label, state=state)


def _pick_insets(before: np.ndarray, after: np.ndarray, count: int = MAX_INSETS) -> list[int]:
    """Indices of the corners worth magnifying: the largest moves, then a typical one."""
    if len(before) == 0:
        return []
    shift = np.linalg.norm(np.asarray(after) - np.asarray(before), axis=1)
    shift = np.where(np.isfinite(shift), shift, -1.0)
    order = list(np.argsort(-shift))
    chosen = order[: max(1, count - 1)]
    median = int(np.argsort(shift)[len(shift) // 2])
    if median not in chosen:
        chosen.append(median)
    return [int(i) for i in chosen[:count]]


# ---------------------------------------------------------------------------------------------
# Reproduced ArUco steps (thresholding, contours, candidates, decoding)
# ---------------------------------------------------------------------------------------------

def _opencv_threshold_windows(params) -> list[int]:
    """The adaptive-threshold window sizes OpenCV's detector tries (aruco_detector.cpp)."""
    low = int(params.adaptiveThreshWinSizeMin)
    high = int(params.adaptiveThreshWinSizeMax)
    step = max(1, int(params.adaptiveThreshWinSizeStep))
    sizes = []
    for i in range((high - low) // step + 1):
        size = low + i * step
        if size % 2 == 0:
            size += 1
        sizes.append(max(3, size))
    return sizes


def _opencv_threshold(gray: np.ndarray, window: int, constant: float) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                 int(window), float(constant))


def _aruco2_threshold(gray: np.ndarray, box: int, thres: int) -> np.ndarray:
    """aruco2's "darker than the local mean" threshold (aruco2.cpp MarkerDetector::detect)."""
    mean = cv2.boxFilter(gray, -1, (int(box), int(box)), normalize=True,
                         borderType=cv2.BORDER_REPLICATE | cv2.BORDER_ISOLATED)
    difference = cv2.subtract(mean, gray)
    _, binary = cv2.threshold(difference, int(thres), 255, cv2.THRESH_BINARY)
    return binary


def _classify_contours(binary: np.ndarray, *, rule: str, params=None, min_size: int = 10,
                       limit: int = MAX_CONTOURS):
    """Reproduce the contour -> square-candidate filter, with a reason for each rejection.

    :param rule: ``"opencv"`` (aruco_detector.cpp ``_findMarkerContours``) or ``"aruco2"``
        (aruco2.cpp ``MarkerDetector::detect``).
    :return: ``(outlines, quads, reasons)`` -- simplified contour outlines (for drawing), and
        for each contour that is not too small, its 4-corner approximation (or None) and the
        verdict ``"kept"`` / a plain-language reason.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    h, w = binary.shape[:2]
    if rule == "opencv":
        min_perimeter = float(params.minMarkerPerimeterRate) * max(w, h)
        max_perimeter = float(params.maxMarkerPerimeterRate) * max(w, h)
        accuracy = float(params.polygonalApproxAccuracyRate)
        corner_rate = float(params.minCornerDistanceRate)
        border = int(params.minDistanceToBorder)
    else:
        min_perimeter, max_perimeter = 4.0 * min_size, float("inf")
        accuracy, corner_rate, border = 0.03, 0.0, 0
    outlines, quads, reasons = [], [], []
    for contour in contours:
        n = len(contour)
        if n < min_perimeter or n > max_perimeter:
            continue  # noise and the frame: not drawn either, there are thousands
        if len(outlines) < limit:
            outline = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2).astype(np.float32)
            outlines.append(outline)
        approx = cv2.approxPolyDP(contour, n * accuracy, True).reshape(-1, 2)
        if len(approx) != 4:
            quads.append(None)
            reasons.append("not four corners")
            continue
        if not cv2.isContourConvex(approx.reshape(-1, 1, 2)):
            quads.append(approx.astype(np.float64))
            reasons.append("not convex")
            continue
        side_sq = np.sum((np.roll(approx, -1, axis=0) - approx).astype(np.float64) ** 2, axis=1)
        if rule == "opencv":
            if side_sq.min() < (n * corner_rate) ** 2:
                quads.append(approx.astype(np.float64))
                reasons.append("corners too close together")
                continue
            if (np.any(approx[:, 0] < border) or np.any(approx[:, 1] < border)
                    or np.any(approx[:, 0] > w - 1 - border) or np.any(approx[:, 1] > h - 1 - border)):
                quads.append(approx.astype(np.float64))
                reasons.append("too close to the image edge")
                continue
        elif side_sq.min() < min_size * min_size:
            quads.append(approx.astype(np.float64))
            reasons.append("a side is too short")
            continue
        quads.append(approx.astype(np.float64))
        reasons.append("kept")
    return outlines, quads, reasons


def _sort_clockwise(corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    d1, d2 = corners[1] - corners[0], corners[2] - corners[0]
    if d1[0] * d2[1] - d1[1] * d2[0] < 0.0:
        corners = corners[[0, 3, 2, 1]]
    return corners


def _read_marker_raw(gray: np.ndarray, corners: np.ndarray, dictionary, *, rule: str,
                     params=None, invert: bool = False) -> tuple[np.ndarray, Optional[np.ndarray], str, int, int]:
    """Warp one candidate flat and read its bit grid, as the detector would.

    :param rule: ``"opencv"`` follows ``_extractBits`` (warp at ``perspectiveRemovePixelPerCell``,
        Otsu unless the cell contrast is below ``minOtsuStdDev``, cell majority vote ignoring
        ``perspectiveRemoveIgnoredMarginPerCell``); ``"aruco2"`` samples each cell centre and
        Otsu-thresholds the samples, as aruco2.cpp does on its first attempt.
    :return: ``(crop, bits, state, marker_id, rotation)``.
    """
    n = int(dictionary.markerSize)
    cells = n + 2
    corners = _sort_clockwise(corners).astype(np.float32)
    square = np.array([[0, 0], [CROP_SIDE - 1, 0], [CROP_SIDE - 1, CROP_SIDE - 1], [0, CROP_SIDE - 1]],
                      dtype=np.float32)
    homography = cv2.getPerspectiveTransform(corners, square)
    source = 255 - gray if invert else gray
    crop = cv2.warpPerspective(gray, homography, (CROP_SIDE, CROP_SIDE), flags=cv2.INTER_LINEAR)
    crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)  # contrast-stretched for display
    if rule == "opencv":
        cell_px = int(params.perspectiveRemovePixelPerCell)
        side = cell_px * cells
        target = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
        warped = cv2.warpPerspective(source, cv2.getPerspectiveTransform(corners, target), (side, side),
                                     flags=cv2.INTER_NEAREST)
        margin = int(float(params.perspectiveRemoveIgnoredMarginPerCell) * cell_px)
        inner = warped[cell_px // 2: side - cell_px // 2, cell_px // 2: side - cell_px // 2]
        if inner.size and float(np.std(inner)) < float(params.minOtsuStdDev):
            return crop, None, "too_uniform", -1, 0
        _, binary = cv2.threshold(warped, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bits = np.zeros((cells, cells), dtype=np.uint8)
        for r in range(cells):
            for c in range(cells):
                cell = binary[r * cell_px + margin:(r + 1) * cell_px - margin,
                              c * cell_px + margin:(c + 1) * cell_px - margin]
                bits[r, c] = 0 if np.count_nonzero(cell) > cell.size / 2 else 1  # 1 = black
        max_border = int(n * n * float(params.maxErroneousBitsInBorderRate))
        correction = float(params.errorCorrectionRate)
    else:
        # aruco2.cpp: up to maxAttemptsPerCandidate (5) readings -- the corners as found, then
        # slightly jittered ones; the third reading thresholds the cells locally instead of with
        # Otsu. The jitter here is seeded (aruco2's own cv::RNG stream is not reachable).
        rng = np.random.default_rng(0)
        first = None
        for attempt in range(5):
            jittered = corners if attempt == 0 else (
                corners + rng.normal(0.0, 0.75, size=corners.shape).astype(np.float32))
            bits = _aruco2_sample_bits(source, jittered, cells, adaptive=(attempt == 2))
            verdict = _identify(bits, dictionary, 0, 0.0)
            if first is None:
                first = (bits, verdict)
            if verdict[0] == "accepted":
                return (crop, bits) + verdict
        bits, verdict = first
        return (crop, bits) + verdict
    return (crop, bits) + _identify(bits, dictionary, max_border, correction)



def _read_marker(gray: np.ndarray, corners: np.ndarray, dictionary, *, rule: str,
                 params=None, invert: bool = False):
    """Warp one candidate flat and read its bit grid, as the detector would.

    The crop is the photo's own grey values (contrast-stretched); ``bits`` use 1 = black *as
    seen in the photo*, so a white-bordered (inverted) marker has a white border in both.
    See :func:`_read_marker_raw` for the reading rules. Returns
    ``(crop, bits, state, marker_id, rotation)``.
    """
    crop, bits, state, marker_id, rotation = _read_marker_raw(
        gray, corners, dictionary, rule=rule, params=params, invert=invert)
    if invert and bits is not None:
        bits = (1 - bits).astype(np.uint8)
    return crop, bits, state, marker_id, rotation


def _aruco2_sample_bits(source: np.ndarray, corners: np.ndarray, cells: int, *,
                        adaptive: bool) -> np.ndarray:
    """Sample each cell centre (bilinear) and threshold the samples, as aruco2.cpp does."""
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(unit, corners.astype(np.float32))
    centres = (np.arange(cells) + 0.5) / cells
    grid = np.stack(np.meshgrid(centres, centres), axis=-1).reshape(-1, 1, 2).astype(np.float32)
    points = cv2.perspectiveTransform(grid, homography).reshape(cells, cells, 2)
    samples = cv2.remap(source.astype(np.float32), points[..., 0].astype(np.float32),
                        points[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    samples = np.clip(samples + 0.5, 0, 255).astype(np.uint8)
    if adaptive:  # thres255Adaptive(off=2, thres=5)
        mean = cv2.boxFilter(samples, -1, (5, 5), normalize=True, borderType=cv2.BORDER_DEFAULT)
        white = (mean.astype(np.int32) - 5) < samples.astype(np.int32)
    else:
        _, binary = cv2.threshold(samples, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        white = binary > 0
    return (~white).astype(np.uint8)  # 1 = black


def _identify(bits: np.ndarray, dictionary, max_border: int, correction: float) -> tuple[str, int, int]:
    """Border check, then dictionary lookup, of a bit grid (1 = black, border included)."""
    border = np.concatenate([bits[0, :], bits[-1, :], bits[1:-1, 0], bits[1:-1, -1]])
    if int(np.sum(border == 0)) > max_border:
        return "bad_border", -1, 0
    # The dictionaries count a white cell as 1 (aruco2.cpp getMarkerId, OpenCV _extractBits).
    inner_bits = np.ascontiguousarray(1 - bits[1:-1, 1:-1]).astype(np.uint8)
    try:
        found, marker_id, rotation = dictionary.identify(inner_bits, correction)
    except cv2.error:
        found, marker_id, rotation = False, -1, 0
    if not found:
        return "not_in_dictionary", -1, 0
    return "accepted", int(marker_id), int(rotation)


def _dictionary_bits(dictionary, marker_id: int) -> Optional[np.ndarray]:
    """Bits of one dictionary entry (n x n, 1 = white), or None when unreadable."""
    try:
        size = int(dictionary.markerSize)
        entry = np.asarray(dictionary.bytesList)[int(marker_id)]
        return np.asarray(cv2.aruco.Dictionary.getBitsFromByteList(
            entry.reshape(1, *entry.shape), size), dtype=np.uint8).reshape(size, size)
    except (cv2.error, IndexError, AttributeError, ValueError, TypeError):
        return None


def _attach_lookup(crop: MarkerCrop, dictionary, correctable: int = 0) -> None:
    """Fill a read crop's ``code``/``lookup`` for the "code -> ID" picture (display only).

    ``crop.bits`` is 1 = black *as seen*; a white-bordered marker is read with light and
    dark swapped, so the code the dictionary sees is the swapped one.
    """
    if crop.bits is None or crop.marker_id < 0:
        return
    seen = np.asarray(crop.bits, dtype=np.uint8)
    raw = (1 - seen) if crop.inverted else seen          # 1 = black in the read polarity
    code = (1 - raw[1:-1, 1:-1]).astype(np.uint8)        # dictionary convention: 1 = white
    own = _dictionary_bits(dictionary, crop.marker_id)
    if own is None or own.shape != code.shape:
        return
    errors = [int(np.sum(np.rot90(code, k) != own)) for k in range(4)]
    turns = int(np.argmin(errors))
    crop.code = np.rot90(code, turns).copy()
    crop.turns = turns
    crop.bit_errors = errors[turns]
    try:
        total = int(np.asarray(dictionary.bytesList).shape[0])
    except (AttributeError, IndexError):
        total = 0
    crop.dictionary_size = total
    # The match, then the closest *other* code in the dictionary (fewest differing cells over
    # its four turns, SCI2-15): that distance is what error correction works within. The
    # other code is stored turned to its closest alignment with the read code.
    crop.lookup = [(int(crop.marker_id), own)]
    crop.correctable_cells = max(int(correctable), 0)
    closest = _closest_other_code(dictionary, crop.code, crop.marker_id, total)
    if closest is not None:
        other, bits, distance, ties = closest
        crop.lookup.append((other, bits))
        crop.closest_distance, crop.closest_ties = int(distance), int(ties)


def _closest_other_code(dictionary, code: np.ndarray, own_id: int, total: int):
    """``(id, bits turned to best match code, cells that differ, how many other entries
    are that close)`` of the dictionary entry nearest to ``code`` other than ``own_id``
    (display only), or None."""
    if not total or code is None:
        return None
    try:
        size = int(dictionary.markerSize)
        bytes_list = np.asarray(dictionary.bytesList)
        bits = np.asarray([cv2.aruco.Dictionary.getBitsFromByteList(
            bytes_list[k].reshape(1, *bytes_list[k].shape), size) for k in range(total)],
            dtype=np.uint8).reshape(total, size, size)
    except (cv2.error, AttributeError, ValueError, TypeError, IndexError):
        return None
    code = np.asarray(code, dtype=np.uint8)
    if code.shape != (size, size):
        return None
    turned = np.stack([np.rot90(code, k) for k in range(4)])          # (4, n, n)
    distance = np.sum(bits[:, None] != turned[None], axis=(2, 3))      # (total, 4)
    best = distance.min(axis=1)
    best[int(own_id)] = 10 ** 6
    other = int(np.argmin(best))
    k = int(np.argmin(distance[other]))
    ties = int(np.sum(best == best[other]))
    # Turn the other entry so it lines up with the code as shown (code = rot90(entry, -k)).
    return other, np.rot90(bits[other], -k).copy(), int(best[other]), ties


_CROP_ORDER = {"accepted": 0, "bad_border": 1, "not_in_dictionary": 2, "too_uniform": 3}

_REASON_WORDS = {
    "bad_border": "its border is not solid black",
    "not_in_dictionary": "its pattern is not one of this target's markers",
    "too_uniform": "too little contrast to read",
}


# ---------------------------------------------------------------------------------------------
# Building the steps
# ---------------------------------------------------------------------------------------------

class _Builder:
    """Assembles :class:`ExplainStep` objects from the detector's records."""

    def __init__(self, photo: np.ndarray, prepared: np.ndarray, *, upscale_factor: int,
                 preparation_scale: float, events: list[dict], target, file_dtype: str,
                 rescale_and_gamma: bool, scale: float, gamma: float,
                 loader: str = UPPER_BYTE):
        self.photo = photo
        self.prepared = prepared
        self.events = events
        self.target = target
        self.file_dtype = file_dtype
        self.rescale_and_gamma = rescale_and_gamma
        self.scale = scale
        self.gamma = gamma
        self.loader = loader
        self.upscale = int(upscale_factor)
        # detector-image pixels per detection-frame pixel
        self.frame_to_prepared = float(preparation_scale)
        # photo pixels per detection-frame pixel
        self.frame_to_photo = 1.0 / float(self.upscale)
        self.steps: list[ExplainStep] = []
        self.gray = _gray(_to_uint8(prepared))
        #: Channels stored in the photo file (1 = a grey file), None when unknown.
        self.file_channels: Optional[int] = None

    # -- common ----------------------------------------------------------------------------

    def add(self, step: ExplainStep) -> ExplainStep:
        self.steps.append(step)
        return step

    def photo_display(self):
        return _display(self.photo, self.frame_to_photo)

    def gray_display(self):
        return _display(self.gray, self.frame_to_prepared)

    def input_steps(self, puzzle: bool) -> None:
        # *puzzle* is kept for callers; the preparation steps now follow the setting alone,
        # because Phase 1 applies "Rescale and apply gamma" to any target it is ticked for.
        photo = np.asarray(self.photo)
        image, scale = self.photo_display()
        depth = {"uint8": 8, "uint16": 16}.get(self.file_dtype, None)
        h, w = photo.shape[:2]
        self.add(ExplainStep(
            key="input", title="Your photo",
            caption="The photo exactly as Phase 1 read it from disk.",
            image=image, image_scale=scale,
            counts={"width px": w, "height px": h,
                    "bit depth": depth if depth else self.file_dtype,
                    "channels": 1 if photo.ndim == 2 else photo.shape[2]},
            extra={"file_dtype": self.file_dtype, "read_dtype": str(photo.dtype),
                   "file_channels": self.file_channels,
                   "grey_content": _is_grey_content(photo)}))
        if self.rescale_and_gamma:
            if self.loader == MIN_MAX and photo.dtype in (np.uint8, np.uint16):
                base = min_max_to_uint8(photo)
                cut = ("16-bit values are stretched from this photo's darkest to its brightest "
                       "value onto 8 bits")
            else:
                base = _to_uint8(photo)
                cut = "16-bit values are cut to 8 bits"
            gray_full = _gray(base)
            gray_image, gray_scale = _display(gray_full, self.frame_to_photo)
            self.add(ExplainStep(
                key="grayscale", title="Grey image",
                caption=f"Colour is dropped and {cut}: only brightness matters.",
                image=gray_image, image_scale=gray_scale,
                counts={"mean brightness": round(float(gray_full.mean()), 1)}))
            prepared_rgb, prepared_scale = _display(self.prepared, self.frame_to_prepared)
            ph, pw = np.asarray(self.prepared).shape[:2]
            size = ("kept at full size" if self.scale == 1.0 else
                    f"shrunk to {self.scale:g}x" if self.scale < 1.0 else
                    f"enlarged to {self.scale:g}x")
            tone = "lifts the dark tones" if self.gamma < 1.0 else "adjusts the tones"
            self.add(ExplainStep(
                key="rescale_gamma", title="Shrink and brighten",
                caption=(f"The photo is {size} and gamma {self.gamma:g} {tone}, "
                         "so the pattern stands out."),
                image=prepared_rgb, image_scale=prepared_scale,
                counts={"scale": self.scale, "gamma": self.gamma, "width px": pw, "height px": ph,
                        "16-bit loader": self.loader},
                extra={"histogram_before": np.bincount(gray_full.reshape(-1), minlength=256),
                       "histogram_after": np.bincount(self.gray.reshape(-1), minlength=256)}))
        else:
            gray_image, gray_scale = self.gray_display()
            self.add(ExplainStep(
                key="grayscale", title="Grey image",
                caption="Colour is dropped: the detector only looks at brightness.",
                image=gray_image, image_scale=gray_scale,
                counts={"mean brightness": round(float(self.gray.mean()), 1)},
                extra={"histogram": np.bincount(self.gray.reshape(-1), minlength=256)}))
        image, scale = self.gray_display()
        ph, pw = self.gray.shape[:2]
        self.add(ExplainStep(
            key="detector_input", title="What the finder gets",
            caption="This exact image is handed to the marker finder.",
            image=image, image_scale=scale,
            counts={"width px": pw, "height px": ph,
                    "upscaled": f"{self.upscale}x" if self.upscale > 1 else "no"}))

    # -- ArUco threshold / contours / candidates / decoding ----------------------------------

    def aruco_front_steps(self, *, rule: str, binaries: list[tuple[str, np.ndarray]],
                          representative: int, threshold_caption: str, threshold_counts: dict,
                          accepted: list[tuple[str, np.ndarray]], dictionaries: list,
                          opencv_params=None, detector_candidates: Optional[list[np.ndarray]] = None,
                          invert_for: Optional[list[bool]] = None, min_size: int = 10,
                          threshold_extra: Optional[dict] = None) -> None:
        """Threshold, contours, square candidates and decoding (reproduced), labelled with the
        markers the real run accepted (*accepted*, detection frame)."""
        f2p = self.frame_to_prepared
        shown, shown_scale = _display(binaries[representative][1], f2p)
        thumbs = [(name, _display(binary, f2p, max_side=480)[0]) for name, binary in binaries]
        self.add(ExplainStep(
            key="threshold", title="Black-and-white image",
            caption=threshold_caption, image=shown, image_scale=shown_scale,
            images=thumbs, counts=threshold_counts, source="reproduced",
            extra=dict(threshold_extra or {}, representative=binaries[representative][0],
                       representative_index=int(representative), rule=rule,
                       display_per_prepared=float(shown_scale / f2p))))

        # contours and the square filter, per binary image (candidates are pooled)
        representative_outlines: list[np.ndarray] = []
        pooled: list[tuple[np.ndarray, str, bool]] = []  # (quad in frame px, verdict, inverted)
        n_contours = 0
        n_four = 0
        n_not_four = 0
        for index, (_name, binary) in enumerate(binaries):
            outlines, quads, reasons = _classify_contours(
                binary, rule=rule, params=opencv_params, min_size=min_size)
            if index == representative:
                representative_outlines = outlines
                n_contours = len(reasons)
                n_four = sum(1 for q in quads if q is not None)
            invert = bool(invert_for[index]) if invert_for else False
            for quad, reason in zip(quads, reasons):
                if quad is None:
                    n_not_four += 1
                    continue
                pooled.append((_to_frame(quad, f2p), reason, invert))
        image, scale = self.gray_display()
        self.add(ExplainStep(
            key="contours", title="Outlines",
            caption="Every outline of a black region is traced; markers are among them.",
            image=image, image_scale=scale, source="reproduced",
            overlays=[Overlay("polylines", "outlines",
                              [_to_frame(o, f2p) for o in representative_outlines])],
            # Counts describe the picture shown (the representative black-and-white image);
            # the other images the detector also traces are counted apart.
            counts={"outlines": n_contours, "four-cornered": n_four,
                    "images traced": len(binaries)},
            extra={"images traced": len(binaries), "representative": binaries[representative][0],
                   "not four-cornered, all images": n_not_four}))

        kept = _dedupe_quads([(q, inv) for q, r, inv in pooled if r == "kept"])
        geometric_rejects = [(q, r) for q, r, _ in pooled if r != "kept"]
        side = _side_px([a for _, a in accepted]) if accepted else _side_px([q for q, _ in kept])
        if detector_candidates is not None:
            # OpenCV's own candidate list (accepted + rejectedImgPoints) is authoritative.
            candidate_quads = [(np.asarray(q, dtype=np.float64).reshape(4, 2), False)
                               for q in detector_candidates]
            candidate_source = "detector"
        else:
            candidate_quads = kept
            candidate_source = "reproduced"
        accepted_polys = [a for _, a in accepted]
        to_accepted = _match_polygons([q for q, _ in candidate_quads], accepted_polys, 0.35 * side)
        # every accepted marker is a candidate, even one the reproduction missed
        matched = set(i for i in to_accepted if i >= 0)
        for j, (label, poly) in enumerate(accepted):
            if j not in matched:
                candidate_quads.append((np.asarray(poly, dtype=np.float64), False))
                to_accepted.append(j)
        states = ["kept" if j >= 0 else "rejected" for j in to_accepted]
        polygons = [q for q, _ in candidate_quads]
        image, scale = self.gray_display()
        # decode every candidate (bounded) for reasons and the gallery
        crops: list[MarkerCrop] = []
        reasons = []
        both = bool(invert_for) and any(invert_for)
        for (quad, inv), j in zip(candidate_quads, to_accepted):
            if j >= 0:
                # Read where the detector says the marker is, in either polarity it searches.
                crop = self._decode(accepted[j][1], dictionaries, rule=rule, params=opencv_params,
                                    inverts=(False, True) if both else (inv,))
                crop.detector_id = accepted[j][0]
            else:
                crop = self._decode(quad, dictionaries, rule=rule, params=opencv_params, inverts=(inv,))
                crop.detector_id = None
            if j >= 0:
                reasons.append("")
            else:
                reasons.append(_REASON_WORDS.get(crop.state, "not read as a marker"))
            crops.append(crop)
        self.add(ExplainStep(
            key="candidates", title="Square candidates",
            caption="Outlines with four straight sides become candidates; the rest are dropped.",
            image=image, image_scale=scale,
            source="mixed" if candidate_source == "detector" else "reproduced",
            overlays=[
                Overlay("polygons", "not square", [q for q, _ in geometric_rejects],
                        states=["dropped"] * len(geometric_rejects),
                        reasons=[r for _, r in geometric_rejects]),
                Overlay("polygons", "candidates", polygons, states=states, reasons=reasons),
            ],
            counts={"candidates": len(polygons), "became markers": int(sum(s == "kept" for s in states)),
                    "rejected": int(sum(s == "rejected" for s in states)),
                    "not square": len(geometric_rejects)},
            extra={"candidate_source": candidate_source}))

        ordered = sorted(crops, key=lambda c: (_CROP_ORDER.get(c.state, 9) if c.detector_id is None else -1,
                                               str(c.detector_id)))[:MAX_CROPS]
        reproduced_ok = sum(1 for c in crops if c.detector_id is not None and c.state == "accepted")
        n_real = sum(1 for c in crops if c.detector_id is not None)
        self.add(ExplainStep(
            key="decoding", title="Reading each square",
            caption="Each candidate is warped flat and its grid of black and white cells is read as an ID.",
            image=None, crops=ordered, source="reproduced",
            counts={"read as an ID": int(sum(c.state == "accepted" for c in crops)),
                    "bad border": int(sum(c.state == "bad_border" for c in crops)),
                    "unknown pattern": int(sum(c.state == "not_in_dictionary" for c in crops)),
                    "too faint": int(sum(c.state == "too_uniform" for c in crops))},
            extra={"reproduced_agrees": f"{reproduced_ok}/{n_real}",
                   "crop_side": CROP_SIDE}))

    def _decode(self, quad: np.ndarray, dictionaries: list, *, rule: str, params,
                inverts: tuple = (False,)) -> MarkerCrop:
        poly_prepared = _to_prepared(quad, self.frame_to_prepared)
        best = None
        for invert in inverts:
            for face, dictionary in dictionaries:
                crop, bits, state, marker_id, rotation = _read_marker(
                    self.gray, poly_prepared, dictionary, rule=rule, params=params, invert=invert)
                item = MarkerCrop(image=crop, bits=bits, marker_bits=int(dictionary.markerSize),
                                  state=state, marker_id=marker_id, rotation=rotation,
                                  polygon=np.asarray(quad, dtype=np.float64), face=face,
                                  inverted=bool(invert))
                if state == "accepted":
                    correctable = 0
                    if rule == "opencv":
                        try:
                            correctable = int(int(dictionary.maxCorrectionBits)
                                              * float(params.errorCorrectionRate))
                        except (AttributeError, TypeError, ValueError):
                            correctable = 0
                    _attach_lookup(item, dictionary, correctable)
                    return item
                if best is None or (best.state == "bad_border" and state != "bad_border"):
                    best = item
        return best

    def markers_step(self, accepted: list[tuple[str, np.ndarray]], caption: str, counts: dict) -> None:
        image, scale = self.gray_display()
        self.add(ExplainStep(
            key="markers", title="Markers found", caption=caption,
            image=image, image_scale=scale,
            overlays=[Overlay("polygons", "markers", [a for _, a in accepted],
                              states=["accepted"] * len(accepted),
                              labels=[label for label, _ in accepted])],
            counts=counts))

    # -- final -----------------------------------------------------------------------------

    def final_step(self, keys: np.ndarray, points: np.ndarray, check: ConsistencyCheck) -> None:
        image, scale = self.photo_display()
        labels = [_key_label(k) for k in keys]
        overlays = [Overlay("points", "detections", np.asarray(points, dtype=np.float64).reshape(-1, 2),
                            states=["accepted"] * len(labels), labels=labels)]
        counts = {"points found": int(len(labels))}
        if check.available:
            counts["match saved run"] = "yes" if check.ok else "no"
            if np.isfinite(check.max_shift_px):
                counts["max shift px"] = round(float(check.max_shift_px), 3)
        self.add(ExplainStep(
            key="final", title="Result",
            caption="These are the points Phase 1 keeps from this photo.",
            image=image, image_scale=scale, overlays=overlays, counts=counts,
            extra={"consistency": check}))


def _dedupe_quads(quads: list[tuple[np.ndarray, bool]]) -> list[tuple[np.ndarray, bool]]:
    """Drop near-identical quads found in several threshold images (keep the first)."""
    kept: list[tuple[np.ndarray, bool]] = []
    centres: list[np.ndarray] = []
    for quad, inv in quads:
        centre = quad.mean(axis=0)
        side = float(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).mean())
        if any(np.linalg.norm(centre - c) < 0.25 * side for c in centres):
            continue
        kept.append((quad, inv))
        centres.append(centre)
    return kept


# ---------------------------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------------------------

def _charuco_nearest_markers(board) -> tuple[np.ndarray, list[list[int]]]:
    """For each ChArUco chessboard corner, the board markers next to it (as OpenCV picks them)."""
    chess = np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)[:, :2]
    obj = [np.asarray(o, dtype=np.float64).reshape(4, 3)[:, :2] for o in board.getObjPoints()]
    centres = np.array([o.mean(axis=0) for o in obj])
    nearest = []
    for corner in chess:
        distance = np.linalg.norm(centres - corner, axis=1)
        nearest.append([int(i) for i in np.flatnonzero(distance <= distance.min() * 1.0001 + 1e-9)])
    return chess, nearest


def _charuco_estimates(board, marker_ids: np.ndarray, marker_corners: list[np.ndarray],
                       corner_ids: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Each corner's position from its neighbouring markers' homographies, before refinement.

    Reproduces OpenCV's local-homography interpolation (charuco_detector.cpp): one homography
    per detected marker, the corner projected through each neighbouring marker's, averaged.
    """
    chess, nearest = _charuco_nearest_markers(board)
    board_ids = [int(i) for i in np.asarray(board.getIds()).reshape(-1)]
    obj = [np.asarray(o, dtype=np.float64).reshape(4, 3)[:, :2] for o in board.getObjPoints()]
    detected = {int(mid): np.asarray(c, dtype=np.float64).reshape(4, 2)
                for mid, c in zip(np.asarray(marker_ids).reshape(-1), marker_corners)}
    homographies = {}
    for index, mid in enumerate(board_ids):
        if mid in detected:
            homographies[index] = cv2.getPerspectiveTransform(
                obj[index].astype(np.float32), detected[mid].astype(np.float32))
    estimates, support = [], []
    for cid in np.asarray(corner_ids).reshape(-1):
        projected, used = [], []
        for index in nearest[int(cid)]:
            if index in homographies:
                point = cv2.perspectiveTransform(chess[int(cid)].reshape(1, 1, 2), homographies[index])
                projected.append(point.reshape(2))
                used.append(board_ids[index])
        estimates.append(np.mean(projected, axis=0) if projected else np.array([np.nan, np.nan]))
        support.append(used)
    return np.asarray(estimates, dtype=np.float64).reshape(-1, 2), support


def _explain_charuco(builder: _Builder, faces: list[dict], aruco2_markers: Optional[list]) -> str:
    """ChArUco boards (ChArUco, Ccube) read with OpenCV's or aruco2's marker finder."""
    target = builder.target
    f2p = builder.frame_to_prepared
    boards = list(getattr(target, "boards", None) or [getattr(target, "board")])
    detectors = getattr(target, "board_detectors", None)
    if detectors is not None and not isinstance(detectors, (list, tuple)):
        detectors = [detectors]
    backend = "aruco2" if aruco2_markers is not None else "aruco1"
    gray = builder.gray
    accepted: list[tuple[str, np.ndarray]] = []
    marker_faces: list[int] = []
    corner_records = []
    multi = len(boards) > 1
    for record in faces:
        face = int(record["face"])
        detail = record["detail"]
        board = boards[face]
        c_corners, c_ids, mloc, mid = detail.get("detect_board", (None, None, None, None))
        if mloc is not None and mid is not None:
            for corners, marker_id in zip(mloc, np.asarray(mid).reshape(-1)):
                label = f"{face}:{int(marker_id)}" if multi else str(int(marker_id))
                accepted.append((label, _to_frame(np.asarray(corners).reshape(4, 2), f2p)))
                marker_faces.append(face)
        if c_ids is not None and c_corners is not None and mloc is not None:
            ids = np.asarray(c_ids).reshape(-1).astype(int)
            after = np.asarray(c_corners, dtype=np.float64).reshape(-1, 2)
            before, support = _charuco_estimates(
                board, np.asarray(mid).reshape(-1), [np.asarray(c).reshape(4, 2) for c in mloc], ids)
            corner_records.append((face, ids, before, after, support))

    # thresholding as the finder did it
    if backend == "aruco1":
        params = detectors[0].getDetectorParameters()
        windows = _opencv_threshold_windows(params)
        binaries = [(f"window {w} px", _opencv_threshold(gray, w, params.adaptiveThreshConstant))
                    for w in windows]
        # OpenCV's own candidates: accepted + rejected, from one detector (geometry does not
        # depend on the dictionary), with Phase 1's own parameters.
        dictionary = boards[0].getDictionary()
        aruco = cv2.aruco.ArucoDetector(dictionary, params)
        found, found_ids, rejected = aruco.detectMarkers(gray)
        candidates = [_to_frame(np.asarray(c).reshape(4, 2), f2p)
                      for c in list(found or []) + list(rejected or [])]
        counts = {"window sizes px": ", ".join(str(w) for w in windows),
                  "constant": float(params.adaptiveThreshConstant)}
        representative = _best_binary(binaries, rule="opencv", params=params)
        builder.aruco_front_steps(
            rule="opencv", binaries=binaries, representative=representative,
            threshold_caption=("Each pixel is compared with the average of its neighbourhood: darker "
                               "turns black. Several neighbourhood sizes are tried."),
            threshold_counts=counts, accepted=accepted,
            dictionaries=[(i, b.getDictionary()) for i, b in enumerate(boards)],
            opencv_params=params, detector_candidates=candidates,
            threshold_extra={"window_sizes": windows,
                             "constant": float(params.adaptiveThreshConstant)})
    else:
        import aruco2 as _aruco2  # noqa: F401  (available: the run itself used it)
        defaults = _aruco2.DetectionParameters()
        binary = _aruco2_threshold(gray, defaults.box_filter_size, defaults.thres)
        from pyCamSet.calibration_targets.markers.aruco2 import resolve_dictionary
        dictionary = resolve_dictionary(int(target._aruco_dict_int), "aruco2")
        global_accepted = [(str(int(m)), _to_frame(np.asarray(c).reshape(4, 2), f2p))
                           for m, c in aruco2_markers]
        builder.aruco_front_steps(
            rule="aruco2", binaries=[(f"window {defaults.box_filter_size} px", binary)],
            representative=0,
            threshold_caption=("Each pixel darker than the average of its neighbourhood (by more than "
                               f"{defaults.thres} grey levels) turns black."),
            threshold_counts={"window px": int(defaults.box_filter_size),
                              "darker by": int(defaults.thres)},
            accepted=global_accepted, dictionaries=[(None, dictionary)],
            min_size=int(defaults.min_size),
            threshold_extra={"window_sizes": [int(defaults.box_filter_size)],
                             "constant": int(defaults.thres)})

    n_faces = len(set(marker_faces))
    if backend == "aruco2":
        shown = global_accepted
        caption = ("The markers whose cells read as one of this target's IDs; "
                   "each face then keeps its own." if multi else
                   "The markers whose cells read as one of this target's IDs.")
    else:
        shown = accepted
        caption = ("The markers whose cells read as one of this target's IDs, per face."
                   if multi else "The markers whose cells read as one of this target's IDs.")
    counts = {"markers": len(shown)}
    if multi:
        counts["faces seen"] = n_faces
    builder.markers_step(shown, caption, counts)

    # corners interpolated between markers, then refined
    image, scale = builder.gray_display()
    labels, before_all, after_all, support_labels = [], [], [], []
    for face, ids, before, after, support in corner_records:
        for cid, b, a, s in zip(ids, before, after, support):
            labels.append(f"{face}:{cid}" if multi else str(cid))
            before_all.append(_to_frame(b, f2p))
            after_all.append(_to_frame(a, f2p))
            support_labels.append(",".join(str(m) for m in s))
    before_all = np.asarray(before_all, dtype=np.float64).reshape(-1, 2)
    after_all = np.asarray(after_all, dtype=np.float64).reshape(-1, 2)
    builder.add(ExplainStep(
        key="board_corners", title="Corners between markers",
        caption="Each chessboard corner is placed from the markers on either side of it.",
        image=image, image_scale=scale, source="mixed",
        overlays=[Overlay("polygons", "markers", [a for _, a in accepted],
                          states=["accepted"] * len(accepted)),
                  Overlay("points", "corners", after_all, states=["kept"] * len(labels),
                          labels=labels)],
        counts={"corners": len(labels)},
        extra={"supporting_markers": dict(zip(labels, support_labels)),
               "estimate_source": "reproduced (marker homographies)"}))
    _subpixel_step(builder, before_all, after_all, labels,
                   accepted_mask=np.ones(len(labels), dtype=bool),
                   caption="Each corner is nudged to the exact crossing of the dark and light squares.",
                   source="mixed")
    return backend


def _best_binary(binaries: list[tuple[str, np.ndarray]], *, rule: str, params) -> int:
    """The threshold image with the most square candidates (the most telling one to show)."""
    best, best_count = 0, -1
    for index, (_, binary) in enumerate(binaries):
        _, _, reasons = _classify_contours(binary, rule=rule, params=params, limit=0)
        count = sum(1 for r in reasons if r == "kept")
        if count > best_count:
            best, best_count = index, count
    return best


def _subpixel_step(builder: _Builder, before: np.ndarray, after: np.ndarray, labels: list[str], *,
                   accepted_mask: np.ndarray, caption: str, source: str,
                   extra: Optional[dict] = None) -> None:
    image, scale = builder.gray_display()
    finite = np.all(np.isfinite(before), axis=1) & np.all(np.isfinite(after), axis=1)
    shift = np.where(finite, np.linalg.norm(after - before, axis=1), np.nan)
    states = ["moved" if ok else "unchanged" for ok in accepted_mask]
    segments = np.stack([before, after], axis=1) if len(before) else np.zeros((0, 2, 2))
    insets = []
    usable = np.flatnonzero(finite)
    side = 8.0
    if len(after) > 1:
        tree = after[usable]
        if len(tree) > 1:
            d = np.linalg.norm(tree[:, None] - tree[None], axis=-1)
            d[d == 0] = np.inf
            side = float(np.median(d.min(axis=1)))
    for local in _pick_insets(before[usable], after[usable]):
        index = int(usable[local])
        inset = _inset(builder.gray, builder.frame_to_prepared, after[index], max(4.0, 0.35 * side),
                       before[index], after[index], title=f"corner {labels[index]}",
                       label=f"{shift[index]:.2f} px", state=states[index])
        if inset is not None:
            insets.append(inset)
    counts = {"corners": int(len(labels)),
              "moved": int(np.sum(accepted_mask)),
              "median move px": round(float(np.nanmedian(shift)), 3) if np.any(finite) else 0.0,
              "largest move px": round(float(np.nanmax(shift)), 3) if np.any(finite) else 0.0}
    builder.add(ExplainStep(
        key="subpixel", title="Sub-pixel refinement", caption=caption,
        image=image, image_scale=scale, source=source,
        overlays=[Overlay("segments", "refinement", segments, states=states, labels=labels),
                  Overlay("points", "before", before, states=["predicted"] * len(labels)),
                  Overlay("points", "after", after, states=states, labels=labels)],
        insets=insets, counts=counts, extra=dict(extra or {}, shift_px=shift)))


def _explain_grid_board(builder: _Builder, faces: list[dict]) -> str:
    """aruco2's own grid-board detector (ChArUco2, Ccube2)."""
    import aruco2 as _aruco2  # noqa: F401  (available: the run itself used it)
    from pyCamSet.calibration_targets.markers.aruco2 import resolve_dictionary
    from pyCamSet.calibration_targets.markers.aruco2_gridboard import (
        _aruco2_erosion_passes, _grid_board_padding)

    target = builder.target
    f2p = builder.frame_to_prepared
    gray = builder.gray
    multi = len(faces) > 1
    # aruco2.cpp detectBWMarkers on the padded image: black markers (box 25, eroded), white
    # markers (box 15, inverted).
    pad = _grid_board_padding(gray.shape[1])
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    black = _aruco2_threshold(padded, 25, 3)
    passes = _aruco2_erosion_passes(padded.shape[1])
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    binaries = []
    for i in range(1, passes):
        eroded = cv2.erode(black, kernel, iterations=i)[pad:-pad, pad:-pad]
        binaries.append((f"black markers, eroded {i}x", eroded))
    white = 255 - _aruco2_threshold(padded, 15, 3)
    white = cv2.erode(white, kernel, iterations=1)[pad:-pad, pad:-pad]
    binaries.append(("white markers (inverted)", white))
    dictionary = resolve_dictionary(int(target._aruco_dict_int), "aruco2")

    accepted: list[tuple[str, np.ndarray]] = []
    for record in faces:
        detail = record["detail"]
        for marker_id, corners in detail.get("markers", []) or []:
            accepted.append((str(int(marker_id)), _to_frame(corners, f2p)))
    builder.aruco_front_steps(
        rule="aruco2", binaries=binaries, representative=0,
        threshold_caption=("Each pixel darker than its neighbourhood average turns black: once to find "
                           "black markers, once inverted to find white ones."),
        threshold_counts={"black window px": 25, "white window px": 15, "darker by": 3,
                          "erosion passes": max(1, passes - 1)},
        accepted=accepted, dictionaries=[(None, dictionary)],
        invert_for=[False] * (len(binaries) - 1) + [True], min_size=10,
        threshold_extra={"window_sizes": [25, 15], "constant": 3, "erosion_passes": passes - 1})

    face_summary = {}
    for record in faces:
        detail = record["detail"]
        face_summary[int(record["face"])] = detail.get("outcome", "not searched")
    found_faces = [f for f, o in face_summary.items() if o == "found"]
    counts = {"markers": len(accepted)}
    if multi:
        counts["faces found"] = len(found_faces)
    builder.markers_step(
        accepted,
        "The markers aruco2 kept: one connected group per face, in the right order.", counts)

    # board corners: aruco2's own corners, the edge cut, supporting markers
    image, scale = builder.gray_display()
    raw_pts, raw_labels, raw_states, raw_support = [], [], [], {}
    for record in faces:
        face, detail = int(record["face"]), record["detail"]
        if "raw_ids" not in detail:
            continue
        grid_w, grid_h = detail["grid_size"]
        ids = detail.get("board_ids") or list(range(grid_w * grid_h))
        found = {int(m) for m, _ in detail.get("markers", [])}
        for cid, point, inside in zip(detail["raw_ids"], detail["raw_points"], detail["inside_margin"]):
            label = f"{face}:{int(cid)}" if multi else str(int(cid))
            raw_pts.append(_to_frame(point, f2p))
            raw_labels.append(label)
            raw_states.append("kept" if inside else "dropped")
            row, col = divmod(int(cid), grid_w + 1)
            touching = [ids[r * grid_w + c] for r in (row - 1, row) for c in (col - 1, col)
                        if 0 <= r < grid_h and 0 <= c < grid_w]
            raw_support[label] = [int(m) for m in touching if int(m) in found]
    builder.add(ExplainStep(
        key="board_corners", title="Corners between markers",
        caption="Where neighbouring markers touch, their shared corner becomes a board corner.",
        image=image, image_scale=scale,
        overlays=[Overlay("polygons", "markers", [a for _, a in accepted],
                          states=["accepted"] * len(accepted)),
                  Overlay("points", "corners", np.asarray(raw_pts).reshape(-1, 2), states=raw_states,
                          labels=raw_labels,
                          reasons=["" if s == "kept" else "too close to the image edge"
                                   for s in raw_states])],
        counts={"corners": len(raw_pts), "too close to edge": int(sum(s == "dropped" for s in raw_states))},
        extra={"supporting_markers": raw_support, "faces": face_summary}))

    # sub-pixel (saddle) refinement
    before, after, labels, moved, symmetry = [], [], [], [], []
    for record in faces:
        face, detail = int(record["face"]), record["detail"]
        refine = detail.get("refine")
        if not refine:
            continue
        for cid, b, s, ok, sym in zip(detail["candidate_ids"], refine["before"], refine["saddle"],
                                      refine["accepted"], refine["symmetry"]):
            labels.append(f"{face}:{int(cid)}" if multi else str(int(cid)))
            before.append(_to_frame(b, f2p))
            after.append(_to_frame(s if ok else b, f2p))
            moved.append(bool(ok))
            symmetry.append(float(sym))
    _subpixel_step(
        builder, np.asarray(before, dtype=np.float64).reshape(-1, 2),
        np.asarray(after, dtype=np.float64).reshape(-1, 2), labels,
        accepted_mask=np.asarray(moved, dtype=bool),
        caption=("Each corner slides to the exact crossing point of the pattern; one that does "
                 "not look like a clean crossing keeps its first position."),
        source="detector", extra={"symmetry": np.asarray(symmetry), "min_symmetry": 0.3})

    # lattice check (validation)
    points, predicted, states, vlabels, residual, tolerance = [], [], [], [], [], []
    for record in faces:
        face, detail = int(record["face"]), record["detail"]
        check = detail.get("validate")
        if not check:
            continue
        unsupported = set(int(i) for i in np.asarray(check.get("unsupported", [])).reshape(-1))
        for index, (cid, point, pred, keep) in enumerate(zip(
                detail["candidate_ids"], check["points"], check["predicted"], check["keep"])):
            vlabels.append(f"{face}:{int(cid)}" if multi else str(int(cid)))
            points.append(_to_frame(point, f2p))
            predicted.append(_to_frame(pred, f2p))
            if keep:
                states.append("kept")
            elif index in unsupported:
                states.append("removed")
            else:
                states.append("rejected")
            residual.append(float(check["residual"][index]))
            tolerance.append(float(check["tolerance"][index]))
    points_arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    predicted_arr = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    reasons = {"kept": "", "removed": "too few neighbours to check",
               "rejected": "disagrees with where its neighbours put it"}
    image, scale = builder.gray_display()
    builder.add(ExplainStep(
        key="grid_check", title="Do the neighbours agree?",
        caption="Each corner's neighbours predict where it should be; a corner far from that is removed.",
        image=image, image_scale=scale,
        overlays=[Overlay("segments", "prediction", np.stack([predicted_arr, points_arr], axis=1)
                          if len(points_arr) else np.zeros((0, 2, 2)), states=states),
                  Overlay("points", "predicted", predicted_arr, states=["predicted"] * len(states)),
                  Overlay("points", "corners", points_arr, states=states, labels=vlabels,
                          reasons=[reasons[s] for s in states])],
        counts={"checked": len(states), "kept": int(sum(s == "kept" for s in states)),
                "removed": int(sum(s != "kept" for s in states))},
        extra={"residual_px": np.asarray(residual) / f2p, "tolerance_px": np.asarray(tolerance) / f2p}))
    return "aruco2_gridboard"


def _explain_puzzleboard(builder: _Builder, detail: dict, cube: bool) -> str:
    f2p = builder.frame_to_prepared
    detector_image = detail.get("detector_image")
    if detector_image is None:
        return "puzzleboard"
    try:
        from puzzle_board.puzzle_board_detector import _filter_image
        from puzzle_board.cornerdetection import sub_pixel_detection
    except ImportError:
        _filter_image = None
    if _filter_image is not None:
        img2, profile, strength, *_ = _filter_image(detector_image)
        dots = np.column_stack(np.where(strength > 0))
        h, w = detector_image.shape[:2]
        dots = dots[(dots[:, 0] > 3) & (dots[:, 1] > 3) & (dots[:, 0] <= h - 3) & (dots[:, 1] <= w - 3)]
        sub = sub_pixel_detection.get_subpixel_positions(profile, dots) if len(dots) else np.zeros((0, 2))
        candidates = _to_frame(np.asarray(sub, dtype=np.float64).reshape(-1, 2)[:, ::-1], f2p)
        smooth, smooth_scale = _display(np.clip(img2 * 255.0, 0, 255).astype(np.uint8), f2p)
        builder.add(ExplainStep(
            key="puzzle_smooth", title="Smoothed grey image",
            caption="The grey image is blurred slightly so that noise does not look like corners.",
            image=smooth, image_scale=smooth_scale, source="reproduced", counts={"blur sigma px": 1}))
        response = strength / (float(np.nanmax(strength)) + 1e-12)
        heat, heat_scale = _display(np.clip(np.sqrt(np.clip(response, 0, 1)) * 255.0, 0, 255).astype(np.uint8), f2p)
        # Display only (ED2-18): the continuous crossing score before the peak picking
        # (the Hessian detector's product of its two corner responses, positive part).
        score_extra = {}
        try:
            from puzzle_board import puzzle_board_detector as _pbd
            _profile, raw_score = _pbd.h_d.detect_corners(img2, k=1.0)
            raw_score = np.clip(np.asarray(raw_score, dtype=np.float64), 0, None)
            top = float(np.percentile(raw_score[raw_score > 0], 99.5)) if np.any(raw_score > 0) else 0.0
            if top > 0:
                score_map, _ = _display(np.clip(np.sqrt(np.clip(raw_score / top, 0, 1)) * 255.0,
                                                0, 255).astype(np.uint8), f2p)
                score_extra = {"score_map": score_map}
        except Exception:  # noqa: BLE001 - the picture then shows the peaks only
            score_extra = {}
        builder.add(ExplainStep(
            key="puzzle_corners", title="Corner candidates",
            caption="Bright spots mark places that look like a chessboard crossing; each becomes a candidate.",
            image=heat, image_scale=heat_scale, source="reproduced",
            overlays=[Overlay("points", "candidates", candidates, states=["candidate"] * len(candidates))],
            counts={"candidates": int(len(candidates))}, extra=score_extra))

    patches = detail.get("patches") or []
    grid_lines, grid_points, patch_sizes = [], [], []
    bit_points, bit_states, bit_values = [], [], []
    decoded_pts, decoded_labels = [], []
    for patch in patches:
        valid = patch.get("valid")
        sub = patch.get("sub_dot")
        if valid is None or sub is None:
            continue
        xy = _to_frame(np.asarray(sub, dtype=np.float64)[..., ::-1], f2p)
        patch_sizes.append(int(valid.sum()))
        rows, cols = valid.shape
        for r in range(rows):
            for c in range(cols):
                if not valid[r, c]:
                    continue
                grid_points.append(xy[r, c])
                if c + 1 < cols and valid[r, c + 1]:
                    grid_lines.append(np.stack([xy[r, c], xy[r, c + 1]]))
                if r + 1 < rows and valid[r + 1, c]:
                    grid_lines.append(np.stack([xy[r, c], xy[r + 1, c]]))
        for kind in ("h", "v"):
            ok = patch.get(f"{kind}valid")
            mid = patch.get(f"{kind}sub_dot")
            correct = patch.get(f"{kind}correct")
            values = patch.get(f"{kind}bits")
            if ok is None or mid is None:
                continue
            mid_xy = _to_frame(np.asarray(mid, dtype=np.float64)[..., ::-1], f2p)
            for index in zip(*np.nonzero(ok)):
                bit_points.append(mid_xy[index])
                good = bool(np.asarray(correct)[index]) if correct is not None else True
                bit_states.append("bit_ok" if good else "bit_wrong")
                try:
                    bit_values.append(int(round(float(np.asarray(values)[index]))))
                except (TypeError, ValueError, IndexError):
                    bit_values.append(0)  # the detector stores bits as -1 / +1; 0 = unknown
        positions = patch.get("positions")
        if positions is not None:
            for r in range(rows):
                for c in range(cols):
                    if valid[r, c]:
                        decoded_pts.append(xy[r, c])
                        decoded_labels.append(f"{int(positions[0, r, c])},{int(positions[1, r, c])}")
    image, scale = builder.gray_display()
    builder.add(ExplainStep(
        key="puzzle_grid", title="Linking corners into a grid",
        caption="Each candidate is linked to its neighbours; linked patches large enough to read are kept.",
        image=image, image_scale=scale,
        overlays=[Overlay("segments", "grid", np.asarray(grid_lines).reshape(-1, 2, 2),
                          states=["kept"] * len(grid_lines)),
                  Overlay("points", "grid points", np.asarray(grid_points).reshape(-1, 2),
                          states=["kept"] * len(grid_points))],
        counts={"patches": len(patch_sizes), "grid points": len(grid_points),
                "min width": detail.get("min_width")}))
    builder.add(ExplainStep(
        key="puzzle_bits", title="Reading the edge bits",
        caption="Between two corners a small dot is light or dark: one bit of the board's position code.",
        image=image, image_scale=scale,
        overlays=[Overlay("points", "edge bits", np.asarray(bit_points).reshape(-1, 2), states=bit_states)],
        counts={"bits read": len(bit_points),
                "agree with code": int(sum(s == "bit_ok" for s in bit_states)),
                "disagree": int(sum(s == "bit_wrong" for s in bit_states))},
        extra={"bit_values": np.asarray(bit_values, dtype=int)}))
    builder.add(ExplainStep(
        key="puzzle_decoded", title="Where on the board?",
        caption="The bits are matched against the printed code, giving every corner its board position.",
        image=image, image_scale=scale,
        overlays=[Overlay("points", "decoded", np.asarray(decoded_pts).reshape(-1, 2),
                          states=["accepted"] * len(decoded_pts), labels=decoded_labels)],
        counts={"positions decoded": len(decoded_pts)}))
    if cube:
        assigned_keys, assigned_pts = detail.get("assigned", ([], []))
        dropped = detail.get("window_dropped", [])
        pts = [_to_frame(p, f2p) for p in assigned_pts]
        pts += [_to_frame(d[2], f2p) for d in dropped]
        labels = [_key_label(k) for k in assigned_keys] + [f"{d[0]},{d[1]}" for d in dropped]
        states = ["accepted"] * len(assigned_pts) + ["dropped"] * len(dropped)
        reasons = [""] * len(assigned_pts) + [
            "between two faces" if d[3] == 0 else "on more than one face" for d in dropped]
        builder.add(ExplainStep(
            key="puzzle_faces", title="Which face?",
            caption="Each board position belongs to one cube face; points in the gaps between faces are dropped.",
            image=image, image_scale=scale,
            overlays=[Overlay("points", "faces", np.asarray(pts).reshape(-1, 2), states=states,
                              labels=labels, reasons=reasons)],
            counts={"assigned": len(assigned_pts), "dropped": len(dropped)}))
        gate_keys, gate_pts = detail.get("after_gate", ([], []))
        final_keys, _ = detail.get("after_duplicates", ([], []))
        final_set = {_key_tuple(k) for k in final_keys}
        dup_pts = [_to_frame(p, f2p) for k, p in zip(gate_keys, gate_pts)
                   if _key_tuple(k) not in final_set]
        dup_labels = [_key_label(k) for k in gate_keys if _key_tuple(k) not in final_set]
        builder.add(ExplainStep(
            key="puzzle_duplicates", title="Twice-found points",
            caption="A point decoded at two places cannot be trusted, so both copies are dropped.",
            image=image, image_scale=scale,
            overlays=[Overlay("points", "duplicates", np.asarray(dup_pts).reshape(-1, 2),
                              states=["dropped"] * len(dup_pts), labels=dup_labels)],
            counts={"dropped": len(dup_pts), "kept": len(final_keys)}))
    return "puzzleboard"


# ---------------------------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------------------------

def _as_detection(saved) -> Optional[ImageDetection]:
    if saved is None:
        return None
    if isinstance(saved, ImageDetection):
        return saved
    if isinstance(saved, Mapping):
        return ImageDetection(saved.get("keys"), saved.get("image_points"))
    keys, points = saved
    return ImageDetection(keys, points)


def check_consistency(keys, points, saved) -> ConsistencyCheck:
    """Compare a live detection with the saved Phase 1 detection of the same photo."""
    live = {_key_tuple(k): np.asarray(p, dtype=np.float64) for k, p in zip(keys, points)}
    saved = _as_detection(saved)
    if saved is None:
        return ConsistencyCheck(False, False, len(live), 0,
                                message="No saved Phase 1 result to compare with.")
    stored = {}
    seen: dict = {}
    if saved.has_data:
        for k, p in zip(saved.keys, saved.image_points):
            stored[_key_tuple(k)] = np.asarray(p, dtype=np.float64).reshape(-1)[-2:]
            seen[_key_tuple(k)] = seen.get(_key_tuple(k), 0) + 1
    duplicates = sorted(k for k, n in seen.items() if n > 1)
    common = sorted(set(live) & set(stored))
    missing = sorted(set(stored) - set(live))
    extra = sorted(set(live) - set(stored))
    shift = max((float(np.linalg.norm(live[k] - stored[k])) for k in common), default=float("nan"))
    ok = not missing and not extra and (not common or shift <= 0.01)
    if ok:
        message = (f"Live recompute matches the saved run: {len(common)} points, "
                   f"largest difference {0.0 if not common else shift:.3g} px.")
    else:
        message = (f"Live recompute differs from the saved run: {len(common)} matched, "
                   f"{len(missing)} missing, {len(extra)} extra"
                   + (f", largest difference {shift:.3g} px." if common else "."))
        if missing and set(missing) <= set(duplicates) and not extra:
            message += (f" The saved run holds {len(missing)} point(s) decoded at two places in "
                        "this photo; today's detector drops such points as ambiguous.")
    return ConsistencyCheck(True, ok, len(live), len(stored), matched=len(common),
                            missing_keys=[list(k) for k in missing], extra_keys=[list(k) for k in extra],
                            max_shift_px=shift, message=message,
                            saved_duplicate_keys=[list(k) for k in duplicates])


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def _is_puzzleboard(target) -> bool:
    return target_family(target) == PUZZLEBOARD


def explain_detection(image, target, *, rescale_and_gamma: bool = False,
                      preprocessing_scale: Optional[float] = None,
                      preprocessing_gamma: Optional[float] = None,
                      saved_detection=None, upscale_factor: int = 1,
                      preprocessing_loader: Optional[str] = None) -> DetectionExplanation:
    """Re-run Phase 1's detection on one photo and explain every step it took.

    :param image: the photo, as a path (read exactly as Phase 1 reads it) or an array already
        read that way (``cv2.IMREAD_COLOR``, or ``cv2.IMREAD_UNCHANGED`` with rescale-and-gamma).
    :param target: the calibration target the run used (e.g. from
        :func:`pyCamSet.workflow.targets.target_of_params`).
    :param rescale_and_gamma: whether Phase 1's "Rescale and apply gamma" ran for this photo.
        It is applied to any target when True; deciding whether a run's saved settings applied
        it (old runs: PuzzleBoard only) is :func:`load_phase1_photo`'s job.
    :param preprocessing_scale: the rescale factor; None takes the target's default
        (PuzzleBoard 0.25, ChArUco family 1.0).
    :param preprocessing_gamma: the gamma; None takes the target's default (0.5).
    :param preprocessing_loader: ``"upper_byte"`` or ``"min_max"``; None takes the target's
        default (PuzzleBoard upper byte, ChArUco family min/max).
    :param saved_detection: the saved Phase 1 result for this photo (an
        :class:`ImageDetection`, a ``(keys, points)`` pair or a mapping with those keys), for the
        consistency check; None skips it.
    :param upscale_factor: Phase 1's upscale factor (default 1).
    :return: the ordered steps, the detection itself and the consistency check.
    """
    started = time.perf_counter()
    defaults = PREPARATION_DEFAULTS[target_family(target)]
    scale = float(defaults.scale if preprocessing_scale is None else preprocessing_scale)
    gamma = float(defaults.gamma if preprocessing_gamma is None else preprocessing_gamma)
    loader = defaults.loader if preprocessing_loader is None else preprocessing_loader
    file_dtype = None
    file_channels = None
    if isinstance(image, (str, Path)):
        path = Path(image)
        photo = read_detection_image(path, rescale_and_gamma)
        if photo is None:
            raise FileNotFoundError(f"Could not read the photo {path}")
        if rescale_and_gamma:
            file_dtype = str(photo.dtype)
            file_channels = 1 if photo.ndim == 2 else int(photo.shape[2])
        else:
            original = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            file_dtype = str(original.dtype) if original is not None else str(photo.dtype)
            if original is not None:
                file_channels = 1 if original.ndim == 2 else int(original.shape[2])
    else:
        photo = np.asarray(image)
    file_dtype = file_dtype or str(photo.dtype)

    events: list[dict] = []
    error = None
    try:
        detection = detect_in_image(
            target, photo, upscale_factor=int(upscale_factor), rescale_and_gamma=rescale_and_gamma,
            preprocessing_scale=scale, preprocessing_gamma=gamma, preprocessing_loader=loader,
            explain=events)
    except (cv2.error, ValueError) as exc:
        # Phase 1 records such a photo as having no detections (find_in_imfolder).
        if isinstance(exc, ValueError) and not _looks_like_native_opencv_error(exc):
            raise
        error = str(exc)
        detection = ImageDetection()
    keys = np.asarray(detection.keys)
    points = np.asarray(detection.image_points, dtype=np.float64).reshape(-1, 2) \
        if detection.has_data else np.zeros((0, 2))

    prepared_event = next((e for e in events if e.get("event") == "prepared_image"), None)
    prepared = prepared_event["image"] if prepared_event else photo
    preparation_scale = prepared_event["preparation_scale"] if prepared_event else 1.0
    builder = _Builder(photo, prepared, upscale_factor=int(upscale_factor),
                       preparation_scale=preparation_scale, events=events, target=target,
                       file_dtype=file_dtype, rescale_and_gamma=rescale_and_gamma,
                       scale=scale, gamma=gamma, loader=loader)
    builder.file_channels = file_channels
    puzzle_events =[e for e in events if e.get("event") == "puzzleboard"]
    builder.input_steps(puzzle=bool(puzzle_events))

    backend = "unknown"
    grid_faces = [e for e in events if e.get("event") == "grid_board_face"]
    charuco_faces = [e for e in events if e.get("event") == "charuco_face"]
    aruco2_event = next((e for e in events if e.get("event") == "aruco2_markers"), None)
    if error is None:
        if grid_faces:
            backend = _explain_grid_board(builder, grid_faces)
        elif charuco_faces:
            markers = aruco2_event["markers"] if aruco2_event else None
            if markers is None:
                inner = [f["detail"].get("aruco2_markers") for f in charuco_faces
                         if f["detail"].get("aruco2_markers") is not None]
                markers = inner[0] if inner else None
            if markers is None and getattr(target, "marker_backend", "aruco1") == "aruco2":
                markers = []
            backend = _explain_charuco(builder, charuco_faces, markers)
        elif puzzle_events:
            backend = _explain_puzzleboard(builder, puzzle_events[0]["detail"],
                                           cube=hasattr(target, "face_origins"))

    check = check_consistency(keys, points, saved_detection)
    builder.final_step(keys, points, check)
    return DetectionExplanation(
        steps=builder.steps, backend=backend, target_type=type(target).__name__,
        keys=keys, image_points=points, consistency=check,
        seconds=time.perf_counter() - started, photo_shape=tuple(np.asarray(photo).shape),
        photo_dtype=file_dtype, error=error)


# ---------------------------------------------------------------------------------------------
# Finding the photo, target and saved result of a Phase 1 run
# ---------------------------------------------------------------------------------------------

@dataclass
class Phase1Photo:
    """Everything :func:`explain_detection` needs for one camera + image of a Phase 1 run."""

    photo_path: Optional[Path]
    camera: str
    image_index: int
    target: Any
    rescale_and_gamma: bool
    preprocessing_scale: float
    preprocessing_gamma: float
    upscale_factor: int
    saved_detection: Optional[ImageDetection]
    cameras: tuple[str, ...]
    image_count: int
    #: ``"manifest"`` (the run recorded which file each index was), ``"listing"`` (today's folder
    #: listing, in Phase 1's own order -- not proof it is the same photo) or ``"unavailable"``.
    mapping: str
    #: True when the manifest's recorded size and modification time still match the file.
    verified: bool
    reason: Optional[str] = None
    #: Set when the saved points were moved from the pre-2026-10-13 divide-by-scale mapping,
    #: or from the earlier pixel-corner frame (see :func:`load_phase1_photo`).
    legacy_mapping_note: Optional[str] = None
    #: True for a ChArUco, Ccube or CIco run saved before pyCamSet recorded the pixel frame of
    #: its corners: made under OpenCV 5 they are in today's pixel-centre frame, made under
    #: OpenCV 4 they are :attr:`pixel_corner_offset` further on, and the run does not say which.
    #: (Runs known to be in the pixel-corner frame are moved by :func:`load_phase1_photo`.)
    charuco_convention_unknown: bool = False
    #: How far, on both axes in detection-frame pixels, a run saved in the pixel-corner frame
    #: sits from today's pixel-centre one: half a pixel, divided by the preparation scale when
    #: the rescale-and-gamma preparation ran (the half pixel was added on the rescaled image).
    pixel_corner_offset: float = 0.5
    #: The OpenCV version the run was detected with, when it recorded one.
    saved_opencv_version: Optional[str] = None
    #: How the preparation reduced 16-bit photos (``"upper_byte"`` or ``"min_max"``).
    preprocessing_loader: str = UPPER_BYTE

    def explain(self) -> DetectionExplanation:
        """Explain this photo with the run's own target and settings."""
        if self.photo_path is None:
            raise FileNotFoundError(self.reason or "photo unavailable")
        explanation = explain_detection(
            self.photo_path, self.target, rescale_and_gamma=self.rescale_and_gamma,
            preprocessing_scale=self.preprocessing_scale, preprocessing_gamma=self.preprocessing_gamma,
            preprocessing_loader=self.preprocessing_loader,
            saved_detection=self.saved_detection, upscale_factor=self.upscale_factor)
        if self.legacy_mapping_note and explanation.consistency.available:
            explanation.consistency.message += " " + self.legacy_mapping_note
        check = explanation.consistency
        if (self.charuco_convention_unknown and check.available and not check.ok
                and self.saved_detection is not None and self.saved_detection.has_data):
            # A uniform offset on both axes is exactly what an OpenCV 4 run saved in the
            # pixel-corner frame carries; anything else is a real difference.
            moved = ImageDetection(
                self.saved_detection.keys,
                np.asarray(self.saved_detection.image_points, dtype=np.float64)
                - self.pixel_corner_offset)
            again = check_consistency(explanation.keys, explanation.image_points, moved)
            if again.ok:
                if self.legacy_mapping_note:
                    again.message += " " + self.legacy_mapping_note
                again.message += (" This run was saved before pyCamSet recorded the pixel frame of "
                                  "its corners, and its points sit exactly "
                                  f"{self.pixel_corner_offset:.3g} px on in both axes (the "
                                  "pixel-corner frame of OpenCV before 4.14); they were moved onto today's pixel-centre "
                                  "frame for comparison. Re-run Phase 1 to save them in it.")
                explanation.consistency = again
        check = explanation.consistency
        if (check.available and not check.ok and self.saved_opencv_version
                and self.saved_opencv_version.split(".")[0] != str(OPENCV_MAJOR)):
            check.message += (f" The run was detected with OpenCV {self.saved_opencv_version} "
                              f"and this check with {OPENCV_VERSION}; the two place sub-pixel "
                              "corners slightly differently, so an exact match is not expected.")
        return explanation


def _run_dir_and_metadata(run) -> tuple[Path, dict]:
    if isinstance(run, (str, Path)):
        run_dir = Path(run)
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        return run_dir, metadata
    run_dir = Path(getattr(run, "run_dir"))
    metadata = getattr(run, "metadata", None)
    if metadata is None:
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    return run_dir, _plain(metadata)


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _load_saved(run_dir: Path, metadata: dict):
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection
    from pyCamSet.workflow.education_trace import load_run_pickle

    candidates = [run_dir / "detected_datapoints.pickle"]
    recorded = (metadata.get("artifacts") or {}).get("detected_datapoints_pickle")
    if recorded:
        path = Path(str(recorded))
        try:
            path.resolve(strict=False).relative_to(run_dir.resolve(strict=False))
            candidates.insert(0, path)
        except (OSError, ValueError):
            pass
    for path in candidates:
        if path.exists():
            payload, error = load_run_pickle(path, (tuple, TargetDetection), expected_run_dir=run_dir)
            if error:
                return None, error
            if isinstance(payload, tuple) and len(payload) == 2:
                payload = payload[0]
            if isinstance(payload, TargetDetection):
                return payload, None
            return None, "the run's pickle holds no detections"
    return None, "this run's detections were not saved"


def saved_image_detection(detections, camera: str, image_index: int) -> Optional[ImageDetection]:
    """One camera + image of a saved :class:`TargetDetection`, as an :class:`ImageDetection`."""
    if detections is None or camera not in list(detections.cam_names):
        return None
    data = detections.get_data()
    if data is None or not len(data):
        return ImageDetection()
    camera_index = list(detections.cam_names).index(camera)
    rows = data[(data[:, 0].astype(int) == camera_index) & (data[:, 1].astype(int) == int(image_index))]
    if not len(rows):
        return ImageDetection()
    keys = rows[:, 2:-2].astype(np.int64)
    if keys.shape[1] == 1:
        keys = keys[:, 0]
    return ImageDetection(keys, rows[:, -2:].astype(np.float64))


def load_phase1_photo(run, camera: str, image_index: int) -> Phase1Photo:
    """Find the photo, target, Phase 1 settings and saved detection for one camera + image.

    :param run: a Phase 1 run directory (holding ``metadata.json``), or an object with
        ``run_dir`` and ``metadata`` (the education view's run context).
    :param camera: the camera folder name.
    :param image_index: the image index (pose number) in Phase 1's own order.
    """
    from pyCamSet.workflow.targets import target_of_params

    run_dir, metadata = _run_dir_and_metadata(run)
    params = metadata.get("params") or {}
    diagnostics = metadata.get("diagnostics") or {}
    target = target_of_params(params)
    # Phase 1's own rule (preparation.py): per-target defaults, explicit values win, and
    # runs saved before the any-target change keep the PuzzleBoard-only gate.
    prep = resolve_preparation(params, target)
    rescale = prep is not None
    defaults = PREPARATION_DEFAULTS[target_family(target)]
    scale = prep.scale if prep is not None else defaults.scale
    gamma = prep.gamma if prep is not None else defaults.gamma
    loader = prep.loader if prep is not None else defaults.loader
    upscale = int(params.get("upscale_factor") or 1)
    detections, reason = _load_saved(run_dir, metadata)
    cameras = tuple(str(c) for c in (diagnostics.get("cam_names")
                                       or (list(detections.cam_names) if detections is not None else [])
                                       or params.get("selected_cameras") or []))
    saved = saved_image_detection(detections, camera, image_index)
    legacy_note = None
    if (rescale and saved is not None and saved.has_data
            and metadata.get("coordinate_mapping") != COORDINATE_MAPPING):
        # Saved before the pixel-centre fix: move the points onto today's mapping so the
        # consistency check compares like with like, and say so.
        shift = legacy_mapping_shift(scale)
        saved = ImageDetection(saved.keys, np.asarray(saved.image_points, dtype=np.float64) + shift)
        legacy_note = (f"This run was saved with the earlier rescale mapping; its points were "
                       f"moved by {shift:+.3g} px to today's pixel-centre mapping for comparison.")
    # Which pixel frame the saved points are in. Runs that record it are pixel-centre, as
    # every detector returns today; PuzzleBoard points always were. Before that, every other
    # target saved them in the pixel-corner frame, half a pixel further on (on the image the
    # detector read, so 0.5 / scale native pixels after a rescale) -- except a ChArUco, Ccube
    # or CIco run older than the record of its convention, whose frame followed OpenCV.
    corner_offset = 0.5 / scale if rescale else 0.5
    frame_unknown = False
    if metadata.get("pixel_convention") != PIXEL_CONVENTION and not _is_puzzleboard(target):
        if (type(target).__name__ in CHARUCO_DETECTOR_TARGETS
                and metadata.get("charuco_corner_convention") is None):
            frame_unknown = True
        elif saved is not None and saved.has_data:
            saved = ImageDetection(saved.keys,
                                   np.asarray(saved.image_points, dtype=np.float64) - corner_offset)
            frame_note = (f"This run was saved in the earlier pixel-corner frame; its points were "
                          f"moved by {-corner_offset:+.3g} px to today's pixel-centre frame for "
                          "comparison.")
            legacy_note = frame_note if legacy_note is None else legacy_note + " " + frame_note

    f_loc = params.get("f_loc")
    subfolder = params.get("subfolder_string")
    photo_path, mapping, verified = None, "unavailable", False
    image_count = int(detections.max_ims) if detections is not None else 0
    if f_loc:
        base = Path(str(f_loc)) / camera
        if subfolder:
            base = base / str(subfolder)
        manifest = diagnostics.get("education_manifest")
        entries = None
        if isinstance(manifest, Mapping) and manifest.get("recordable") is True:
            entries = (manifest.get("cameras") or {}).get(camera)
        if isinstance(entries, list) and entries:
            image_count = max(image_count, len(entries))
            if 0 <= image_index < len(entries):
                entry = entries[image_index]
                candidate = (base / str(entry.get("name"))).resolve()
                try:
                    candidate.relative_to(base.resolve())
                    mapping = "manifest"
                    if candidate.is_file():
                        photo_path = candidate
                        stat = candidate.stat()
                        verified = (stat.st_size == entry.get("size")
                                    and stat.st_mtime_ns == entry.get("mtime_ns"))
                        if not verified:
                            reason = "the photo on disk has changed since Phase 1 ran"
                    else:
                        reason = f"photo {entry.get('name')} is no longer on disk"
                except ValueError:
                    reason = "the recorded photo name points outside the camera folder"
            else:
                reason = f"image {image_index} is outside this run's {len(entries)} photos"
        else:
            try:
                listing = enumerate_detection_images(base, params.get("n_lim"))
            except OSError:
                listing = []
            image_count = max(image_count, len(listing))
            if 0 <= image_index < len(listing):
                photo_path, mapping = Path(listing[image_index]), "listing"
                if reason is None:
                    reason = "this run did not record its photo list; today's folder listing is used"
            else:
                reason = f"no photo {image_index} in {base}"
    else:
        reason = "the run does not record its image folder"
    return Phase1Photo(
        photo_path=photo_path, camera=camera, image_index=int(image_index), target=target,
        rescale_and_gamma=rescale, preprocessing_scale=scale, preprocessing_gamma=gamma,
        upscale_factor=upscale, saved_detection=saved, cameras=cameras, image_count=image_count,
        mapping=mapping, verified=verified, reason=reason, legacy_mapping_note=legacy_note,
        preprocessing_loader=loader,
        charuco_convention_unknown=frame_unknown, pixel_corner_offset=corner_offset,
        saved_opencv_version=metadata.get("opencv_version"))
