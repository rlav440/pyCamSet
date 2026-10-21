"""Qt-free data and teaching copy for the Phase 1 ("find the target") education figures.

:mod:`pyCamSet.workflow.education_detect_explain` (a live, checked re-run of Phase 1's own
detection on one photo); this module only chooses *what to show* of it and says it in plain words:

* :func:`split_steps` - which explanation steps belong to stage 2 ("What the detector sees") and
  which to stage 3 ("Finding markers"), with short names for the step pills;
* :func:`focus_box` - one close-up square on the board (a marker and its neighbours), kept the
  same through every step so the learner watches one spot change;
* :func:`step_caption`, :func:`step_chips` - the caption and the stat chips of a step, per level;
* :func:`busiest_image`, :func:`points_per_image` - from the run's own saved detections;
* :func:`coverage_grid`, :func:`hull_polygon` - the per-camera point density and the convex hull
  whose area *is* the saved D1.6 coverage (``workflow/phase1._spatial_coverage``);
* :func:`read_display_photo` - a photo read exactly as Phase 1 reads it, shrunk for display;
* round-1 review: :func:`threshold_demo` (one global cut-off vs the detector's local test, with
  one example pixel), :func:`target_summary` (the run's target faces and who saw which face),
  :func:`decode_counts` (one verdict per square, so chips and bars add up), :func:`um_per_px`
  and :func:`corner_edges` (the sub-pixel close-ups).

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

#: Steps drawn in stage 2 (the preprocessing strip), in the order they happen.
STAGE2_KEYS = ("input", "grayscale", "rescale_gamma", "detector_input", "threshold",
               "puzzle_smooth")

#: Short names for the stage-3 step pills (1-2 words, so 8 pills fit at 1280 px).
SHORT_NAMES = {
    "contours": "Outlines", "candidates": "Squares", "decoding": "Read IDs",
    "markers": "Markers", "board_corners": "Corners", "subpixel": "Sub-pixel",
    "grid_check": "Grid check", "final": "Result",
    "puzzle_corners": "Corner spots", "puzzle_grid": "Grid", "puzzle_bits": "Edge bits",
    "puzzle_decoded": "Positions", "puzzle_faces": "Faces", "puzzle_duplicates": "Duplicates",
}

#: Plain-language L1 caption of each stage-3 step (<= 2 short sentences).
_L1 = {
    "contours": ("The black-and-white image is traced into outlines. Every marker is one of "
                 "these shapes, hidden among a lot of clutter."),
    "candidates": ("Outlines with four corners that look like a square become candidates. The "
                   "candidates that turn out to be markers are in colour; the rest are greyed out."),
    "decoding": ("Each square is flattened and its cells are read as black or white (A). The "
                 "inner cells form a code; looking it up in the target's list gives the ID (B)."),
    "markers": "These squares were read as this target's markers, each with its own ID.",
    "board_corners": ("Where two markers meet, their shared corner becomes a board point. These "
                      "corners are what the calibration uses."),
    "subpixel": ("Each corner slides to the exact crossing of its dark and light squares. "
                 "Ring: first position; dot: refined position."),
    "grid_check": ("Neighbours predict each corner (+); the ring is where it was found. A ring "
                   "outside its dashed circle is removed."),
    "final": "These are the points Phase 1 keeps from this photo.",
    "puzzle_corners": ("The finder scores every pixel for how much it looks like a chessboard "
                       "crossing. Bright spots become corner candidates (rings)."),
    "puzzle_grid": ("Neighbouring corners are linked into a grid. Patches too small to read are "
                    "dropped."),
    "puzzle_bits": ("Between two corners the board prints a small dark or light dot: one bit of a "
                    "position code. Filled squares show the bit read; a red edge disagrees."),
    "puzzle_decoded": ("The bits are matched against the printed code. That tells every corner "
                       "where it sits on the board."),
    "puzzle_faces": ("Each board position belongs to one face of the cube. Points in the gaps "
                     "between faces are dropped."),
    "puzzle_duplicates": ("A point decoded at two places cannot be trusted. Both copies are "
                          "dropped."),
}

#: What L2 adds: how to read the extra panels.
_L2 = {
    "contours": ("C shows the black-and-white image they were traced in; the detector traces "
                 "each of its black-and-white images (dark and light markers) the same way."),
    "candidates": "The bars count what happened to every four-cornered outline.",
    "decoding": ("The outer ring must be one solid colour; greyed tiles were not markers and "
                 "✕ says why."),
    "markers": "Each marker is labelled with its ID (face:marker on a cube).",
    "board_corners": "Crosses mark corners dropped because they lie too close to the image edge.",
    "subpixel": "Each close-up is one corner, contrast-stretched; the arrow joins its two positions.",
    "grid_check": ("A circle is larger where the squares are larger or the neighbours disagree "
                   "among themselves; C compares every corner's miss with its circle."),
    "final": "Labels in the close-up are the point IDs the calibration matches across cameras.",
    "puzzle_corners": "The close-up lays the crossing score over the grey image.",
    "puzzle_grid": "Lines join corners the finder linked as neighbours.",
    "puzzle_bits": "Dark squares are bits read as dark, light squares as light.",
    "puzzle_decoded": "Labels in the close-up are the decoded board positions (row, column).",
    "puzzle_faces": "In the close-up each label is face:point: the cube face, then the point on it.",
    "puzzle_duplicates": "Red crosses would mark points found twice.",
}

#: Steps whose picture is reproduced (OpenCV / compiled internals report nothing back).
REPRODUCED_NOTE = "reproduced with the same OpenCV calls and this run's parameters"


@dataclass(frozen=True)
class StepSplit:
    """The explanation's steps for stage 2 and stage 3."""

    prep: tuple[Any, ...]
    find: tuple[Any, ...]


def split_steps(explanation: Any) -> StepSplit:
    """Stage-2 steps (preprocessing) and stage-3 steps (finding), each in detector order."""
    steps = list(getattr(explanation, "steps", ()) or ())
    prep = tuple(s for s in steps if s.key in STAGE2_KEYS)
    find = tuple(s for s in steps if s.key not in STAGE2_KEYS)
    return StepSplit(prep, find)


def short_name(key: str, title: str = "") -> str:
    """Pill name of a stage-3 step."""
    return SHORT_NAMES.get(key, title or key)


def prep_strip(steps: Sequence[Any]) -> list[Any]:
    """The pictures of the stage-2 strip: photo, grey, (shrink + gamma), detector image.

    ``detector_input`` is dropped when it is the grey image unchanged (no upscale, no rescale),
    so the strip never shows the same picture twice.
    """
    by_key = {s.key: s for s in steps}
    strip = [by_key[k] for k in ("input", "grayscale", "rescale_gamma") if k in by_key]
    det = by_key.get("detector_input")
    if det is not None:
        upscaled = str(det.counts.get("upscaled", "no")) not in ("no", "1x")
        if upscaled:
            strip.append(det)
    for key in ("threshold", "puzzle_smooth"):
        if key in by_key:
            strip.append(by_key[key])
    return strip


# ---------------------------------------------------------------------------------------------
# Captions and chips
# ---------------------------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return "–"
        if v == int(v) and (abs(v) >= 1 or v == 0):
            return f"{int(v):,}"
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"
    return str(value)


#: L1 second sentence when a removal step removed nothing in this photo.
_NONE_HERE = {
    "grid_check": ("Neighbours predict each corner (+); the ring is where it was found. All "
                   "rings lie inside their circles: none removed."),
    "puzzle_faces": ("Each board position belongs to one face of the cube. No point fell in a "
                     "gap between faces in this photo."),
    "puzzle_duplicates": ("A point decoded at two places cannot be trusted, so both copies would "
                          "be dropped. None was found twice in this photo."),
}


def step_caption(step: Any, level: int, *, consistency: Any = None, backend: str = "") -> str:
    """Caption of a stage-3 step: <= 2 short sentences at L1, + 1 at L2, + 1 at L3."""
    text = _L1.get(step.key, step.caption)
    none_key = {"puzzle_duplicates": "dropped", "grid_check": "removed",
                "puzzle_faces": "dropped"}.get(step.key)
    if none_key is not None and not step.counts.get(none_key) and step.key in _NONE_HERE:
        text = _NONE_HERE[step.key]
    if step.key == "final" and consistency is not None and getattr(consistency, "available", False):
        if consistency.ok:
            shift = consistency.max_shift_px if np.isfinite(consistency.max_shift_px) else 0.0
            text += (" Re-running the detector on this photo matches the saved result (same "
                     f"points, largest difference {shift:.2f} px).")
        else:
            text += (" Re-running the detector here does not match the saved result; the saved "
                     "points are the ones Phase 1 used.")
    if level >= 2 and step.key in _L2:
        extra = _L2[step.key]
        if step.key == "decoding":
            extra = decoding_polarity_sentence(step) or extra
        duplicates = list(getattr(consistency, "saved_duplicate_keys", []) or [])
        if step.key == "final" and consistency is not None and not consistency.ok \
                and duplicates:
            n = len(duplicates)
            extra = (f"The saved run holds {n} point{'s' if n != 1 else ''} decoded at two "
                     "places in this photo; today's detector drops such points.")
        text += " " + extra
    if level >= 3:
        if step.source in ("reproduced", "mixed"):
            text += " " + reproduced_sentence(backend)
        else:
            text += " Picture and numbers are from the run."
    return text


def reproduced_sentence(backend: str = "") -> str:
    """How a reproduced picture was made (SCI2-14: aruco2's own pipeline also cuts its
    white-marker image diagonally, which the reproduction does not draw)."""
    # UX3-10: plain words here; the aruco2 detail (no diagonal cuts drawn) is in Numbers.
    return "Redrawn with the detector's own steps; IDs and points are the real run's."



def decoding_polarity_sentence(step: Any) -> str:
    """One sentence on white-bordered (inverted) markers when this photo has them."""
    read = [c for c in getattr(step, "crops", ()) or () if c.detector_id is not None]
    white = [c for c in read if getattr(c, "inverted", False)]
    if not white:
        return ""
    if len(white) == len(read):
        return ("This board prints its markers light on dark (a white ring), so light and dark "
                "are swapped before the lookup.")
    black = [str(c.detector_id) for c in read if not getattr(c, "inverted", False)]
    which = ", ".join(black[:3]) + ("…" if len(black) > 3 else "")
    return (f"Most markers here are printed light on dark (white ring) and are read with light "
            f"and dark swapped; ID {which} is printed dark on light.")


#: Decoding verdicts, in gallery/chip order: (key, chip label, tile tag).
DECODE_CATEGORIES = (
    ("marker", "markers read", ""),
    ("not_kept", "read, not kept", "not kept"),
    ("bad_border", "bad border", "border"),
    ("not_in_dictionary", "unknown code", "unknown"),
    ("too_uniform", "too faint", "faint"),
)


def decode_category(crop: Any) -> str:
    """The verdict of one flattened square, by the real run first.

    ``marker``: the detector kept it; ``not_kept``: the reproduction reads an ID but the
    detector did not keep a marker there; otherwise the reading's own failure.
    """
    if crop.detector_id is not None:
        return "marker"
    if crop.state == "accepted":
        return "not_kept"
    return crop.state if crop.state in ("bad_border", "not_in_dictionary", "too_uniform") \
        else "not_in_dictionary"


def decode_counts(step: Any) -> dict[str, int]:
    """How many squares fell in each :data:`DECODE_CATEGORIES` verdict (all squares)."""
    counts = {key: 0 for key, _l, _t in DECODE_CATEGORIES}
    for crop in getattr(step, "crops", ()) or ():
        counts[decode_category(crop)] += 1
    return counts


#: Chip labels (count key -> short label) and which counts are worth a chip.
_CHIP_LABELS = {
    "outlines": "outlines in this image", "images traced": "black-and-white images",
    "candidates": "candidates",
    "became markers": "markers", "rejected": "dropped", "not square": "not square",
    "read as an ID": "read as ID", "bad border": "bad border",
    "unknown pattern": "unknown", "too faint": "too faint", "markers": "markers",
    "faces found": "faces", "faces seen": "faces", "corners": "corners",
    "too close to edge": "at edge", "moved": "moved", "median move px": "median move px",
    "largest move px": "largest move px", "checked": "checked", "kept": "kept",
    "removed": "removed", "points found": "points", "patches": "patches",
    "grid points": "grid points", "bits read": "bits read", "agree with code": "agree",
    "disagree": "disagree", "positions decoded": "decoded", "assigned": "assigned",
    "dropped": "dropped",
}
_BAD_IF_POSITIVE = {"removed", "disagree", "dropped", "bad border"}


def step_chips(step: Any, limit: int = 4) -> list[tuple[str, str, Optional[str]]]:
    """``(value, label, tone)`` for a step's counts, most telling first, at most ``limit``.

    The squares steps use one funnel so their numbers add up across steps and panels:
    outlines -> four-cornered (step 1); candidates -> markers + dropped (step 2, where
    "dropped" = candidates not read as markers + four-cornered shapes that are not square);
    and the decoding verdicts of every candidate (step 3).
    """
    c = step.counts
    if step.key == "candidates":
        kept = int(c.get("became markers", 0) or 0)
        dropped = int(c.get("rejected", 0) or 0) + int(c.get("not square", 0) or 0)
        return [(_fmt(c.get("candidates", 0)), "square candidates", None),
                (_fmt(kept), "became markers", "accent"),
                (_fmt(dropped), "dropped", "warn" if dropped else None)][:limit]
    if step.key == "decoding" and getattr(step, "crops", None):
        counts = decode_counts(step)
        chips = []
        for key, label, _tag in DECODE_CATEGORIES:
            n = counts[key]
            if key == "marker":
                chips.append((_fmt(n), label, "accent"))
            elif n:
                chips.append((_fmt(n), label, "warn"))
        if len(chips) > limit:
            rest = sum(int(v.replace(",", "")) for v, _l, _t in chips[limit - 1:])
            chips = chips[:limit - 1] + [(_fmt(rest), "other rejects", "warn")]
        return chips
    chips = []
    for key, value in step.counts.items():
        if key not in _CHIP_LABELS or key in ("match saved run", "max shift px"):
            continue
        tone = None
        if key in ("became markers", "markers", "read as an ID", "points found",
                   "positions decoded", "corners", "kept", "grid points"):
            tone = "accent"
        try:
            if key in _BAD_IF_POSITIVE and float(value) > 0:
                tone = "warn"
        except (TypeError, ValueError):
            pass
        chips.append((_fmt(value), _CHIP_LABELS[key], tone))
    return chips[:limit]


def consistency_chip(check: Any) -> Optional[tuple[str, str, str, str]]:
    """``(value, label, tone, tooltip)`` for the live-vs-saved check, or None."""
    if check is None or not getattr(check, "available", False):
        return None
    if check.ok:
        return (f"✓ {check.matched}", "points match saved run", "good", check.message)
    # Say what differs on the chip itself (ED2-18), not only in the tooltip.
    ambiguous = len(list(getattr(check, "saved_duplicate_keys", []) or []))
    missing = len(list(getattr(check, "missing_keys", []) or []))
    extra = len(list(getattr(check, "extra_keys", []) or []))
    if ambiguous and ambiguous >= missing and not extra:
        label = f"saved points re-found · {ambiguous} ambiguous"
        tip = (check.message + " Phase 1 used the saved points; a point decoded at two places "
               "is dropped by today's detector, the rest agree.")
    else:
        label = "saved points re-found"
        tip = check.message + " Phase 1 used the saved points; the pictures show today's re-run."
    return (f"{check.matched} / {check.saved_count}", label, "warn", tip)


# ---------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------

def _polygons(step: Any, name: str, state: Optional[str] = None) -> list[np.ndarray]:
    for overlay in getattr(step, "overlays", ()) or ():
        if overlay.name == name and overlay.kind == "polygons":
            states = overlay.states or [None] * len(overlay)
            return [np.asarray(p, dtype=float).reshape(-1, 2)
                    for p, s in zip(overlay.data, states) if state is None or s == state]
    return []


def _points(step: Any, name: Optional[str] = None) -> np.ndarray:
    for overlay in getattr(step, "overlays", ()) or ():
        if overlay.kind == "points" and (name is None or overlay.name == name) and len(overlay):
            return np.asarray(overlay.data, dtype=float).reshape(-1, 2)
    return np.zeros((0, 2))


def focus_box(explanation: Any, frame_size: tuple[float, float]) -> tuple[float, float, float, float]:
    """A square close-up (x0, y0, x1, y1) in detection-frame pixels, shared by every step.

    ArUco targets: the accepted marker nearest the middle of all markers, with room for its
    neighbours (so a shared corner is inside). PuzzleBoard: the decoded point nearest the middle,
    with about two grid squares around it. Otherwise the middle of the photo.
    """
    width, height = float(frame_size[0]), float(frame_size[1])
    centre, side = None, None
    markers = None
    for step in getattr(explanation, "steps", ()) or ():
        if step.key == "markers":
            markers = _polygons(step, "markers")
    if markers:
        centres = np.array([m.mean(axis=0) for m in markers])
        middle = np.median(centres, axis=0)
        pick = int(np.argmin(np.linalg.norm(centres - middle, axis=1)))
        marker = markers[pick]
        marker_side = float(np.mean(np.linalg.norm(np.roll(marker, -1, axis=0) - marker, axis=1)))
        centre, side = centres[pick], 2.9 * marker_side
    else:
        points = np.asarray(getattr(explanation, "image_points", np.zeros((0, 2))), dtype=float)
        points = points.reshape(-1, 2) if points.size else np.zeros((0, 2))
        if len(points) >= 3:
            middle = np.median(points, axis=0)
            pick = int(np.argmin(np.linalg.norm(points - middle, axis=1)))
            others = np.delete(points, pick, axis=0)
            spacing = float(np.sort(np.linalg.norm(others - points[pick], axis=1))[0])
            centre, side = points[pick], max(5.0 * spacing, 24.0)
    if centre is None:
        centre, side = np.array([width / 2, height / 2]), 0.3 * min(width, height)
    side = float(min(max(side, 24.0), min(width, height)))
    x0 = float(np.clip(centre[0] - side / 2, 0, max(width - side, 0)))
    y0 = float(np.clip(centre[1] - side / 2, 0, max(height - side, 0)))
    return x0, y0, x0 + side, y0 + side


def frame_size(explanation: Any) -> tuple[float, float]:
    """(width, height) of the detection frame (photo pixels x the upscale factor)."""
    for step in getattr(explanation, "steps", ()) or ():
        if step.image is not None and step.image_scale:
            h, w = step.image.shape[:2]
            return w / float(step.image_scale), h / float(step.image_scale)
    shape = getattr(explanation, "photo_shape", (0, 0))
    return float(shape[1]), float(shape[0])


def select_crops(crops: Sequence[Any], capacity: int) -> list[Any]:
    """Crops for the gallery: the detector's markers first, but always room for failures."""
    def natural(crop) -> tuple:
        parts = str(crop.detector_id).split(":")
        return tuple(int(p) if p.lstrip("-").isdigit() else 0 for p in parts)
    real = sorted((c for c in crops if c.detector_id is not None), key=natural)
    failed = [c for c in crops if c.detector_id is None]
    if len(real) + len(failed) <= capacity:
        return list(real) + list(failed)
    keep_failed = min(len(failed), max(capacity // 3, capacity - len(real)))
    keep_real = capacity - keep_failed
    # A spread of failure kinds: round-robin over the states.
    by_state: dict[str, list] = {}
    for crop in failed:
        by_state.setdefault(crop.state, []).append(crop)
    spread: list = []
    while len(spread) < keep_failed and any(by_state.values()):
        for state in list(by_state):
            if by_state[state] and len(spread) < keep_failed:
                spread.append(by_state[state].pop(0))
    return list(real[:keep_real]) + spread


# ---------------------------------------------------------------------------------------------
# Thresholding, shown with one pixel (stage 2)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ThresholdDemo:
    """Global vs local thresholding of this photo, and one pixel's local test.

    Images are in the threshold step's display pixels; ``local`` is the detector's own
    black-and-white image (True = black, i.e. traced). ``probe`` is (x, y) of the pixel
    whose neighbourhood is drawn; ``window`` its side in display pixels.
    """

    grey: np.ndarray
    local: np.ndarray
    global_black: np.ndarray
    cutoff: float
    window: int
    window_prepared: int
    constant: float
    probe: tuple[int, int]
    centre_value: int
    mean_value: float
    local_black: bool
    global_says_black: bool
    difference: np.ndarray      # neighbourhood mean - pixel, every pixel (display px)
    dim_share_lost: float       # share of the dim half's pixels the global cut-off blackens
    #: Of the dim half's *light* pixels (the ones the local rule keeps white, i.e. the light
    #: squares), the share the global cut-off turns black (SCI2-13: the honest loss; dark
    #: pixels are black under both rules and do not count).
    dim_light_lost: float = 0.0


def _box_mean(grey: np.ndarray, window: int) -> np.ndarray:
    import cv2
    return cv2.boxFilter(grey.astype(np.float32), -1, (int(window), int(window)), normalize=True,
                         borderType=cv2.BORDER_REPLICATE)


def threshold_demo(explanation: Any) -> Optional[ThresholdDemo]:
    """Global (Otsu) vs the detector's local threshold on this photo, plus one probe pixel
    where they disagree and whose local verdict matches the detector's image. None when the
    detector has no black-and-white step (PuzzleBoard)."""
    import cv2
    by_key = {s.key: s for s in getattr(explanation, "steps", ()) or ()}
    threshold, grey_step = by_key.get("threshold"), by_key.get("detector_input") or \
        by_key.get("grayscale")
    if threshold is None or threshold.image is None or grey_step is None or grey_step.image is None:
        return None
    grey = np.asarray(grey_step.image)
    if grey.ndim == 3:
        grey = cv2.cvtColor(grey, cv2.COLOR_RGB2GRAY)
    local = np.asarray(threshold.image) > 127
    if grey.shape[:2] != local.shape[:2]:
        grey = cv2.resize(grey, (local.shape[1], local.shape[0]), interpolation=cv2.INTER_AREA)
    grey = np.ascontiguousarray(grey.astype(np.uint8))
    extra = threshold.extra
    windows = list(extra.get("window_sizes") or [25])
    rep = int(extra.get("representative_index", 0) or 0)
    window_prepared = int(windows[rep] if extra.get("rule") == "opencv" and rep < len(windows)
                          else windows[0])
    window = max(3, int(round(window_prepared * float(extra.get("display_per_prepared", 1.0)))))
    window += 1 - window % 2
    constant = float(extra.get("constant", 3) or 0)
    cutoff, _ = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    global_black = grey <= cutoff
    mean = _box_mean(grey, window)
    difference = mean - grey.astype(np.float32)
    computed = difference >= constant if extra.get("rule") == "opencv" else difference > constant
    h, w = grey.shape
    margin = window
    inner = np.zeros_like(local)
    inner[margin:h - margin, margin:w - margin] = True
    agree = inner & (computed == local)
    disagree = agree & (local != global_black)
    box = focus_box(explanation, frame_size(explanation))
    scale = float(threshold.image_scale or 1.0)
    cx, cy = (box[0] + box[2]) / 2 * scale, (box[1] + box[3]) / 2 * scale
    # Prefer a pixel where the global cut-off is wrong, near the close-up, a clear decision.
    candidates = np.argwhere(disagree)
    if not len(candidates):
        candidates = np.argwhere(agree)
    if not len(candidates):
        candidates = np.array([[int(h // 2), int(w // 2)]])
    distance = np.hypot(candidates[:, 1] - cx, candidates[:, 0] - cy)
    clarity = np.abs(difference[candidates[:, 0], candidates[:, 1]] - constant)
    score = distance / max(w, h) - 0.004 * np.minimum(clarity, 20)
    py, px = (int(v) for v in candidates[int(np.argmin(score))])
    # The dim half: pixels darker than the photo's median brightness, by neighbourhood mean.
    dim = mean < np.median(mean)
    lost = float(np.mean(global_black[dim])) if dim.any() else 0.0
    dim_light = dim & ~local
    light_lost = float(np.mean(global_black[dim_light])) if dim_light.any() else 0.0
    return ThresholdDemo(grey=grey, local=local, global_black=global_black, cutoff=float(cutoff),
                         window=window, window_prepared=window_prepared, constant=constant,
                         probe=(px, py), centre_value=int(grey[py, px]),
                         mean_value=float(mean[py, px]), local_black=bool(local[py, px]),
                         global_says_black=bool(global_black[py, px]), difference=difference,
                         dim_share_lost=lost, dim_light_lost=light_lost)


def threshold_layer_name(name: str) -> tuple[str, str]:
    """Plain title and subtitle for one of the detector's black-and-white images."""
    text = str(name)
    if text.startswith("black markers"):
        passes = "".join(ch for ch in text if ch.isdigit()) or "1"
        times = "once" if passes == "1" else f"{passes} times"
        return "Dark markers", f"black areas thinned {times}, so touching squares separate"
    if text.startswith("white markers"):
        return "Light markers", "light and dark swapped"
    if text.startswith("window"):
        return text[:1].upper() + text[1:], "neighbourhood size used"
    return text[:1].upper() + text[1:], ""


# ---------------------------------------------------------------------------------------------
# The target (stage 1) and physical scale
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetSummary:
    """The run's target as the learner sees it: faces, design corners, who saw which face."""

    name: str
    faces: tuple[np.ndarray, ...]     # design corners per face, mm, (n, 3)
    size_mm: float                    # largest extent of the design
    spacing_mm: float                 # typical corner spacing on a face
    seen: np.ndarray                  # (n_cameras, n_faces) photos in which a camera saw a face
    textures: tuple[Optional[np.ndarray], ...]
    #: Every design corner by its key, mm, (n_faces, n_points, 3); NaN where a face has no
    #: corner at that index (the detected keys index into it).
    design: Optional[np.ndarray] = None
    #: Per face: the printed texture's four image corners (top-left, top-right, bottom-right,
    #: bottom-left) placed on the target, mm (4, 3), or None when the target does not say
    #: where its texture sits (ED2-25: the drawing shows the printed pattern).
    texture_corners: tuple[Optional[np.ndarray], ...] = ()

    @property
    def pattern_mm(self) -> float:
        """Largest extent of the printed corner grid on one face (mm)."""
        spans = [float(np.ptp(f, axis=0).max()) for f in self.faces if len(f) > 1]
        return max(spans) if spans else float("nan")


def _unit_to_mm(data: np.ndarray) -> float:
    """Design points are stored in m (Ccube) or mm."""
    return 1000.0 if np.nanmax(np.abs(data)) < 1.0 else 1.0


def _design_mm(target: Any) -> Optional[np.ndarray]:
    """Every design corner by key, mm, (n_faces, n_points, 3) (NaN rows kept)."""
    data = np.asarray(getattr(target, "point_data", np.zeros((0, 3))), dtype=float)
    if data.size == 0:
        return None
    if data.ndim == 2:
        data = data[None]
    data = data.reshape(data.shape[0], -1, data.shape[-1])
    return data * _unit_to_mm(data)


def _points_mm(target: Any) -> tuple[np.ndarray, ...]:
    data = _design_mm(target)
    if data is None:
        return ()
    return tuple(np.asarray(face[np.all(np.isfinite(face), axis=1)], dtype=float)
                 for face in data)


def _texture_corners(target: Any, design: Optional[np.ndarray]) -> tuple[Optional[np.ndarray], ...]:
    """Where each face texture's image corners sit on the target (mm), for targets that place
    their faces from face-local coordinates (Ccube, Ccube2: ``faceData.face_local_coords`` in
    the texture's own frame, x right and y down, ``length`` = the texture's side).

    The face-local -> design map is fitted per face (an exact affine map for a rigid face) and
    refused when it does not reproduce the design to 1e-6 of the face size, so a target whose
    texture sits elsewhere is never drawn wrongly."""
    textures = getattr(target, "textures", None)
    face_data = getattr(target, "faceData", None)
    local = getattr(face_data, "face_local_coords", None)
    length = getattr(target, "length", None)
    if textures is None or local is None or design is None or not length:
        return ()
    local = np.asarray(local, dtype=float)
    if local.ndim != 3 or local.shape[:2] != design.shape[:2]:
        return ()
    out: list[Optional[np.ndarray]] = []
    for k in range(design.shape[0]):
        pts, uv = design[k], local[k, :, :2]
        ok = np.all(np.isfinite(pts), axis=1) & np.all(np.isfinite(uv), axis=1)
        if ok.sum() < 3 or k >= len(textures):
            out.append(None)
            continue
        a = np.column_stack([uv[ok], np.ones(int(ok.sum()))])
        solution, *_ = np.linalg.lstsq(a, pts[ok], rcond=None)
        error = np.abs(a @ solution - pts[ok]).max()
        size = float(np.ptp(pts[ok], axis=0).max()) or 1.0
        if error > 1e-6 * max(size, 1.0) + 1e-9:
            out.append(None)
            continue
        side = float(length)
        corners = np.array([[0, 0, 1], [side, 0, 1], [side, side, 1], [0, side, 1]], dtype=float)
        out.append(corners @ solution)
    return tuple(out)


def _typical_spacing(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float).reshape(-1, points.shape[-1] if points.ndim > 1 else 2)
    if len(points) < 2:
        return float("nan")
    d = np.linalg.norm(points[:, None] - points[None], axis=-1)
    d[d == 0] = np.inf
    return float(np.median(d.min(axis=1)))


def _board_texture(target: Any, design: Optional[np.ndarray]):
    """A flat OpenCV ChArUco board's printed image and where its corners sit (mm), when its
    design corners are the board's own chessboard corners (origin at the top-left square,
    x right, y down, as ``generateImage`` draws it); else None."""
    board = getattr(target, "board", None)
    if board is None or design is None or not hasattr(board, "generateImage"):
        return None
    try:
        n_x, n_y = (int(v) for v in board.getChessboardSize())
        square = float(board.getSquareLength())
        corners = np.asarray(board.getChessboardCorners(), dtype=float).reshape(-1, 3)
    except Exception:  # noqa: BLE001 - not an OpenCV board
        return None
    pts = np.asarray(design[0], dtype=float)
    if pts.shape != corners.shape:
        return None
    unit = float(np.nanmax(np.abs(pts))) / max(float(np.nanmax(np.abs(corners))), 1e-12)
    if not np.allclose(pts, corners * unit, atol=1e-6 * max(unit, 1.0)):
        return None
    width, height = n_x * square, n_y * square
    px_per = 600.0 / max(width, height)
    image = np.asarray(board.generateImage((int(round(width * px_per)),
                                            int(round(height * px_per)))))
    placed = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]],
                      dtype=float) * unit
    return image, placed


def target_summary(target: Any, detections: Any) -> Optional[TargetSummary]:
    """Faces and design corners of the run's target, and how often each camera saw each face."""
    if target is None:
        return None
    design = _design_mm(target)
    faces = _points_mm(target)
    if not faces:
        return None
    every = np.concatenate([f for f in faces if len(f)]) if faces else np.zeros((0, 3))
    size = float(np.ptp(every, axis=0).max()) if len(every) else float("nan")
    spacing = float(np.nanmedian([_typical_spacing(f) for f in faces if len(f) > 1])) \
        if any(len(f) > 1 for f in faces) else float("nan")
    n_cams = len(getattr(detections, "cam_names", []) or [])
    seen = np.zeros((n_cams, len(faces)), dtype=int)
    data = detections.get_data() if detections is not None and hasattr(detections, "get_data") \
        else None
    if data is not None and len(data) and len(faces) > 1 and data.shape[1] >= 6:
        cams, ims, face = (data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2].astype(int))
        ok = (cams >= 0) & (cams < n_cams) & (face >= 0) & (face < len(faces))
        pairs = np.unique(np.stack([cams[ok], ims[ok], face[ok]], axis=1), axis=0)
        np.add.at(seen, (pairs[:, 0], pairs[:, 2]), 1)
    elif data is not None and len(data) and n_cams:
        cams, ims = data[:, 0].astype(int), data[:, 1].astype(int)
        for cam in range(n_cams):
            seen[cam, 0] = len(np.unique(ims[cams == cam]))
    textures: list[Optional[np.ndarray]] = []
    stored = getattr(target, "textures", None)
    for k in range(len(faces)):
        try:
            texture = np.asarray(stored[k]) if stored is not None else None
        except (IndexError, TypeError):
            texture = None
        textures.append(texture if texture is not None and texture.ndim >= 2 else None)
    try:
        placed = _texture_corners(target, design)
    except Exception:  # noqa: BLE001 - the drawing then shows plain plates
        placed = ()
    if not placed and len(faces) == 1 and textures and textures[0] is None:
        board = _board_texture(target, design)
        if board is not None:
            textures[0], corners = board
            placed = (corners,)
    return TargetSummary(name=type(target).__name__, faces=faces, size_mm=size,
                         spacing_mm=spacing, seen=seen, textures=tuple(textures),
                         design=design, texture_corners=placed)


def point_faces(keys: Any, n_faces: int) -> np.ndarray:
    """Face index of each detected point (its first key column on a many-faced target)."""
    keys = np.asarray(keys)
    if keys.size == 0:
        return np.zeros(0, dtype=int)
    keys = keys.reshape(len(keys), -1)
    if n_faces > 1 and keys.shape[1] >= 2:
        return keys[:, 0].astype(int)
    return np.zeros(len(keys), dtype=int)


def um_per_px(points_px: Any, spacing_mm: float = float("nan"), *, keys: Any = None,
              design: Optional[np.ndarray] = None) -> float:
    """About how many µm on the target one photo pixel spans, free of foreshortening (SCI2-12).

    Each face seen in the photo is fitted with the affine map from its design corners (mm, in
    the face's own plane) to their detected pixels. A tilted face is shortened along one
    direction only; the map's larger singular value is the scale along the direction that is
    *not* shortened (px per mm; for a telecentric lens exactly its magnification). The face
    with the most corners decides. Needs the detected ``keys`` and the target's ``design``
    (:attr:`TargetSummary.design`); without them NaN (no µm figure is shown): the old
    nearest-neighbour estimate read about 20 % high on tilted faces. ``spacing_mm`` is
    accepted for older callers and not used.
    """
    del spacing_mm
    points = np.asarray(points_px, dtype=float).reshape(-1, 2)
    if keys is None or design is None or len(points) < 4:
        return float("nan")
    design = np.asarray(design, dtype=float)
    if design.ndim != 3:
        return float("nan")
    keys = np.asarray(keys)
    if keys.size == 0 or len(keys) != len(points):
        return float("nan")
    keys = keys.reshape(len(keys), -1).astype(int)
    if design.shape[0] > 1 and keys.shape[1] >= 2:
        faces, index = keys[:, 0], keys[:, 1]
    else:
        faces, index = np.zeros(len(keys), dtype=int), keys[:, -1]
    best: tuple[int, float] = (0, float("nan"))
    for face in np.unique(faces):
        pick = (faces == face) & (face >= 0) & (face < design.shape[0])
        pick &= (index >= 0) & (index < design.shape[1])
        if pick.sum() < 4:
            continue
        xyz = design[face, index[pick]]
        px = points[pick]
        ok = np.all(np.isfinite(xyz), axis=1) & np.all(np.isfinite(px), axis=1)
        if ok.sum() < 4:
            continue
        xyz, px = xyz[ok], px[ok]
        centre = xyz.mean(axis=0)
        _u, sv, vt = np.linalg.svd(xyz - centre)
        if sv.size < 2 or sv[1] <= 1e-9 * max(sv[0], 1e-12):
            continue  # the corners lie on a line: no 2D scale
        uv = (xyz - centre) @ vt[:2].T
        a = np.column_stack([uv, np.ones(len(uv))])
        solution, *_ = np.linalg.lstsq(a, px, rcond=None)
        scale = float(np.linalg.svd(solution[:2].T, compute_uv=False)[0])  # px per mm
        if scale > 0 and len(px) > best[0]:
            best = (len(px), 1000.0 / scale)
    return best[1]


def corner_name(label: str, n_faces: int = 1) -> str:
    """``"1:25"`` -> ``"F2 · corner 25"`` (faces counted from 1 with their square tag name,
    like every face in every phase, ED2-8)."""
    parts = str(label).replace("corner ", "").split(":")
    if len(parts) == 2 and all(p.strip().lstrip("-").isdigit() for p in parts):
        return f"F{int(parts[0]) + 1} · corner {int(parts[1])}"
    return f"corner {parts[-1].strip()}" if parts else str(label)


def corner_edges(image: np.ndarray, point: Any, radius: float = 4.0) -> Optional[np.ndarray]:
    """The two edge directions crossing at a chessboard corner (unit vectors, 2 x 2).

    Near an X-corner brightness behaves like ``u * v`` along the two edges, so the local
    second derivatives have their eigen-directions on the diagonals between the edges; the
    edges are the sum and difference of those. None when the crop is too flat to tell.
    """
    import cv2
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    smooth = cv2.GaussianBlur(img, (0, 0), max(radius / 2.5, 1.0))
    x0, y0 = float(point[0]), float(point[1])
    r = int(max(3, round(radius)))
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
    xi, yi = np.rint(x0 + xs).astype(int), np.rint(y0 + ys).astype(int)
    ok = (xi >= 0) & (yi >= 0) & (xi < img.shape[1]) & (yi < img.shape[0])
    if ok.sum() < 12:
        return None
    dx, dy = (xi - x0)[ok], (yi - y0)[ok]
    z = smooth[yi[ok], xi[ok]]
    a = np.stack([np.ones_like(dx), dx, dy, dx * dx, dx * dy, dy * dy], axis=1)
    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
    hessian = np.array([[2 * coef[3], coef[4]], [coef[4], 2 * coef[5]]])
    values, vectors = np.linalg.eigh(hessian)
    if values[0] * values[1] >= 0 or not np.all(np.isfinite(vectors)):
        return None  # not a saddle
    e1, e2 = vectors[:, 0], vectors[:, 1]
    edges = np.stack([(e1 + e2) / np.sqrt(2), (e1 - e2) / np.sqrt(2)])
    return edges


# ---------------------------------------------------------------------------------------------
# Saved-run helpers
# ---------------------------------------------------------------------------------------------

def points_per_image(detections: Any) -> np.ndarray:
    """(n_images, n_cameras) count of saved points, from the run's own detections."""
    try:
        n_cams = len(detections.cam_names)
        n_ims = int(detections.max_ims)
    except (AttributeError, TypeError, ValueError):
        return np.zeros((0, 0), dtype=int)
    counts = np.zeros((max(n_ims, 0), n_cams), dtype=int)
    data = detections.get_data() if hasattr(detections, "get_data") else None
    if data is None or not len(data):
        return counts
    cams = data[:, 0].astype(int)
    ims = data[:, 1].astype(int)
    ok = (cams >= 0) & (cams < n_cams) & (ims >= 0) & (ims < n_ims)
    np.add.at(counts, (ims[ok], cams[ok]), 1)
    return counts


def busiest_image(detections: Any, camera: Optional[int] = None) -> int:
    """The image index with the most saved points (0 if none).

    With ``camera``, an image that is busy both for that camera and for the whole rig (equal
    weight to each, as a share of its best image), so the photo grid and the step-by-step
    stages both open on a photo with something to show.
    """
    counts = points_per_image(detections)
    if not counts.size or counts.sum() == 0:
        return 0
    total = counts.sum(axis=1).astype(float)
    if camera is not None and 0 <= camera < counts.shape[1] and counts[:, camera].any():
        own = counts[:, camera].astype(float)
        return int(np.argmax(own / own.max() + total / total.max()))
    return int(np.argmax(total))


def camera_points(detections: Any, camera: str, image_index: Optional[int] = None) -> np.ndarray:
    """Saved points (detection frame) of one camera, for one image or for all images."""
    if detections is None or camera not in list(getattr(detections, "cam_names", [])):
        return np.zeros((0, 2))
    data = detections.get_data()
    if data is None or not len(data):
        return np.zeros((0, 2))
    index = list(detections.cam_names).index(camera)
    rows = data[data[:, 0].astype(int) == index]
    if image_index is not None:
        rows = rows[rows[:, 1].astype(int) == int(image_index)]
    return np.asarray(rows[:, -2:], dtype=float)


def coverage_grid(points: np.ndarray, width: float, height: float,
                  cells_long: int = 24) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Point density on a grid over the image: (counts[rows, cols], extent for imshow)."""
    width, height = max(float(width), 1.0), max(float(height), 1.0)
    if width >= height:
        nx, ny = cells_long, max(2, int(round(cells_long * height / width)))
    else:
        ny, nx = cells_long, max(2, int(round(cells_long * width / height)))
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    counts, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=(ny, nx),
                                  range=((0, height), (0, width)))
    return counts, (0.0, width, height, 0.0)


def hull_polygon(points: np.ndarray) -> Optional[np.ndarray]:
    """Convex hull of the points (the area D1.6 measures), or None."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
    except Exception:
        return None
    return points[hull.vertices]


def _reduction_for(size: Optional[tuple[int, int]], max_side: int) -> int:
    """The largest of 2, 4, 8 whose reduced read still has its longest side at or above
    ``max_side`` (so the final resize only ever shrinks); 1 means a full read."""
    if not size:
        return 1
    longest = max(size)
    for factor in (8, 4, 2):
        if longest / factor >= max_side:
            return factor
    return 1


def read_display_photo(path: Optional[Path], max_side: int = 640) -> Optional[np.ndarray]:
    """The photo read as Phase 1 reads it (8-bit colour), as RGB, shrunk for display.

    A large photo is decoded at 1/2, 1/4 or 1/8 size by OpenCV's reduced read, so the full
    frame is never held for display. Returns None when the file cannot be read. Never writes.
    """
    if path is None:
        return None
    import cv2
    from pyCamSet.calibration_targets.core.abstract_target import read_detection_image
    try:
        factor = _reduction_for(photo_size(Path(path)), max_side)
        if factor > 1:
            image = cv2.imread(str(path), getattr(cv2, f"IMREAD_REDUCED_COLOR_{factor}"))
        else:
            image = read_detection_image(Path(path), False)
    except (OSError, ValueError, cv2.error):
        return None
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
    longest = max(image.shape[:2])
    if longest > max_side:
        factor = max_side / float(longest)
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


def photo_size(path: Optional[Path]) -> Optional[tuple[int, int]]:
    """(width, height) of a photo in its own pixels, read from the file header when possible."""
    if path is None:
        return None
    try:
        from PIL import Image
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def rate_band(rate: float) -> str:
    """The detection summary's own grading of a rate (setup_reports): good >= 90 %, warn >= 60 %."""
    if not np.isfinite(rate):
        return "bad"
    return "good" if rate >= 0.9 else ("warn" if rate >= 0.6 else "bad")


def finite_mapping(values: Any, cameras: Sequence[str]) -> list[float]:
    """Per-camera floats from a saved mapping, NaN where missing."""
    if not isinstance(values, Mapping):
        return [float("nan")] * len(cameras)
    out = []
    for name in cameras:
        value = values.get(name)
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


__all__ = [
    "STAGE2_KEYS", "SHORT_NAMES", "REPRODUCED_NOTE", "StepSplit", "split_steps", "short_name",
    "prep_strip", "step_caption", "step_chips", "consistency_chip", "focus_box", "frame_size",
    "select_crops", "points_per_image", "busiest_image", "camera_points", "coverage_grid",
    "hull_polygon", "read_display_photo", "photo_size", "rate_band", "finite_mapping",
    # round-1 review
    "decoding_polarity_sentence", "DECODE_CATEGORIES", "decode_category", "decode_counts",
    "ThresholdDemo", "threshold_demo", "threshold_layer_name", "TargetSummary",
    "target_summary", "point_faces", "um_per_px", "corner_name", "corner_edges",
]
