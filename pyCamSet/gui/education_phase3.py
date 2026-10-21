'''Teach Phase 3 (all cameras together: bundle adjustment) visually, on the run's own data.

Stages:

1. Who saw what - the pose graph seen from above: camera badges, grey lines for shared photos
   (1-6 px, fewest to most), the first guess's joins as accent arrows with square #k tags, built
   up with the shared transport bar. L2 adds the shared-photo matrix, L3 the corners raster.
2. First guess - ONE target in the rig (the fixed photo's frame; corners coloured by their mean
   error over all photos on the shared log scale) with the cameras around it and the chosen
   photo's target as a ghost outline; beside it the chosen camera's photo with the detected
   corners (rings) and where the first guess predicts them (dots). L2 adds, for runs made before
   the 2026-10-08 seed fix, the first guess's own score; L3 every observation's error.
3. Single-face photos (telecentric runs that report them) - the tilt ambiguity (A one camera,
   B two cameras) and this run's candidate fits.
4. Watching the solver - the recorded accepted steps in 3D (ScenePlayer paced by the error
   change): cameras and photo poses move with their changes exaggerated xN (labelled) and leave
   trails, the target's corners recolour; the chosen camera's photo shows the predicted dots
   sliding onto the found rings; the error curve has a zoom on the fine-tuning steps. L2 adds
   every photo's target centre and per-camera bars; L3 the Schur picture: the normal matrix
   (camera blocks, couplings, one 6x6 block per photo) and the camera system left to solve.
5. Lens fit - what the lens bending Phase 3 adds (distortion, telecentric eps) does to this
   run's corners: arrows on the photo and each camera's error with the bending off and on
   (round 2: the effect, not parameter bars). L2: the same photos in Phase 4, where the target
   is refined too (the bending partly made up for the target's shape); L3: eps in 1/m, Phase 3
   vs Phase 4, and where the model bends around (the principal point, the image of the optical axis).
6. Result - the target in the rig with each camera's mean error, the error ladder (RMS per
   camera alone in Phase 2 -> together in Phase 3 -> Phase 4, one statistic throughout), why
   the cameras must agree (each camera's own pose of one photo vs the shared one) and the
   per-photo strip plot. L2 adds the agreement spreads (turn only for telecentric rigs; the
   position level scaled to the target otherwise), L3 the density of every remaining error.

The saved-evidence logic (provenance, derivation checks, agreement rule, quality and next step)
is unchanged from the reviewed storyboard; it feeds chips, short captions and the collapsed
Numbers drawer instead of paragraphs.
'''
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from pyCamSet.gui import education_perf as perf
from matplotlib.figure import Figure
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from pyCamSet.gui import education_style as es
from pyCamSet.gui.education_levels import compose
from pyCamSet.gui.education_photo_cache import ByteLRU
from pyCamSet.gui.education_stage import Chip, EducationStageView
from pyCamSet.gui.education_worker import EducationWorker
from pyCamSet.workflow.education_playback import build_phase3_frames
from pyCamSet.workflow.education_trace import read_trace
from pyCamSet.workflow.run_quality import blocking_reasons
from pyCamSet.workflow.education_derive import (
    MIN_OUTLIER_N, OUTLIER_DECLARED_N, OUTLIER_MULTIPLIER,
    DerivationUnavailable, derive_phase3,
)


def _plain(value: Any) -> Any:
    """Make the dialog's immutable JSON metadata safe for project helpers."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _number(value: Any, digits: int = 3) -> str:
    """Format finite values without inventing a zero for missing evidence."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return "unavailable"
    return f"{result:.{digits}g}" if np.isfinite(result) else "unavailable"


def _px(value: Any) -> str:
    """A pixel value for chips and captions: 2 decimals below 10, else 3 significant digits."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return "—"
    if not np.isfinite(result):
        return "—"
    return f"{result:.2f}" if abs(result) < 10 else f"{result:.3g}"


def _status_copy(context: Any) -> str:
    """State the recorded disposition and avoid treating an error-free run as a pass."""
    metadata = _plain(context.metadata)
    diagnostics = metadata.get("diagnostics") or {}
    promoted = diagnostics.get("promoted_from") or metadata.get("promoted_from")
    context = getattr(context, "run_context", context)
    if context.status == "failed":
        error = str(metadata.get("error") or "details unavailable").rstrip().rstrip(".")
        return f"Quality check: failed. Run {context.run_id} records an error: {error}."
    if promoted:
        return (f"Created from an Optimisation-tab trial; the normal Phase 3 quality check was not run. "
                f"Only the final error was recorded. Run {context.run_id} is not shown as a normal quality pass.")
    if context.quality_check != "checked":
        return f"Quality check: not checked for run {context.run_id}; its saved record has no checkable quality result."
    report = metadata.get("report") or {}
    termination = (report.get("termination_reason") or report.get("solver_status")
                   or diagnostics.get("termination_reason") or "not recorded")
    if "ftol" in str(termination).lower():  # ED-18: say what the solver's code means
        termination = f"stopped because the steps became tiny ({termination})"
    # The gate writes 'blocking_flags'; the shared helper reads report and gate alike.
    flags = blocking_reasons(metadata)
    return (f"Quality check: {context.status}. Termination: {termination}. "
            f"Warnings or blocking flags: {', '.join(map(str, flags)) if flags else 'none recorded'}. "
            "A lower residual alone does not establish a better calibration.")


def _next_step_copy(context: Any, agreement: str | None = None) -> str:
    """A1 section 9: end the quality frame on one status-keyed recovery or next step.

    ``agreement`` is the overall answer of the 'Did the cameras agree?' check,
    so a complete run does not say 'continue' right after a No.
    """
    metadata = _plain(context.metadata)
    diagnostics = metadata.get("diagnostics") or {}
    promoted = diagnostics.get("promoted_from") or metadata.get("promoted_from")
    context = getattr(context, "run_context", context)
    if context.status == "failed":
        return "Next: fix the recorded error above, then run Phase 3 again."
    if promoted:
        return ("Next: for a normal quality check, run Phase 3 itself from a traceable Phase 2 result, "
                "then continue from that run.")
    if context.quality_check != "checked":
        return ("Next: run Phase 3 again to get a checkable quality result before relying on this run "
                "or continuing to Phase 4.")
    flags = blocking_reasons(metadata)
    if flags:
        # The quality gate's flags name solver-level problems, not photos, so
        # point at what they name and the phases where inputs are fixed.
        return ("Next: fix what the flags above name. If a camera has too few usable points, recheck "
                "its photos in Phase 1 or Phase 2; then run Phase 3 again before continuing to Phase 4.")
    if context.status != "complete":
        return ("Next: the quality check did not record a pass; run Phase 3 again before continuing "
                "to Phase 4.")
    if agreement == "No":
        return ("Next: the cameras did not clearly agree (see the previous stage); review that "
                "before continuing to Phase 4 or exporting this result.")
    return "Next: continue to Phase 4, or export this result."


#: "Did the cameras agree?" warning levels on the RMS spread of each camera's first-guess
#: centre and turn relative to camera 1. The position level is 10 mm, but never more than
#: POSITION_LEVEL_SHARE of the target's size (SCI2-5: a 10 mm limit judges nothing on a 10 mm
#: cube); for telecentric cameras the position is not judged at all (depth is not seen).
POSITION_LEVEL_MM = 10.0
POSITION_LEVEL_SHARE = 0.10
ROTATION_LEVEL_DEG = 5.0
_TELECENTRIC_POSITION_WHY = (
    "telecentric cameras do not see how far away the target is, so each photo's position "
    "along a camera's axis is not measured and the centre spread mixes it in")

_SINGLE_FACE_TITLE = "Single-face images (telecentric)"

#: L3 method copy for the single-face stage (kept whole in the Numbers drawer).
_SINGLE_FACE_METHOD = (
    "Method: for each camera that saw the photo, a closed-form orthographic (affine) fit of the face "
    "gives two starting poses, the two mirror tilts about that camera's viewing direction. This is the "
    "planar pose ambiguity of affine cameras, known as the Necker reversal in weak-perspective pose "
    "estimation (Hartley and Zisserman, Multiple View Geometry, 2nd ed.: section 6.3 for the affine "
    "camera, chapter 14 for the reversal ambiguity). Each start is moved "
    "into the shared frame through that camera's estimated orientation and refined, rotation and full "
    "translation, against every camera's view of the photo at once; the depth one telecentric camera "
    "cannot see is fixed by the others. For the second figure each start is also fitted with its tilt "
    "held and only its position free, which is where the mirror tilt shows its misfit. The best fit "
    "is kept only when at least two cameras saw the photo and its RMS error is at most 20 px, the "
    "limit a single camera's pose is refused at. Limitation: an "
    "accepted fit rests on the cameras fitting one pose together; it is not checked against an "
    "independent reference.")

#: Stage names (short pills). The single-face stage is present only when relevant.
STAGE_GRAPH, STAGE_FIRST, STAGE_SINGLE, STAGE_SOLVER, STAGE_LENS, STAGE_RESULT = (
    "graph", "first", "single", "solver", "lens", "result")
STAGE_TITLES = {
    STAGE_GRAPH: "Who saw what", STAGE_FIRST: "First guess", STAGE_SINGLE: "One face",
    STAGE_SOLVER: "The solver", STAGE_LENS: "Lens fit", STAGE_RESULT: "Result",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nice(value: float) -> float:
    """A round upper limit (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8 x 10^n) at or above ``value``."""
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    for step in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if step * 10 ** exponent >= value:
            return float(step * 10 ** exponent)
    return float(10 ** (exponent + 1))

# ======================================================================================
# Figures (matplotlib, foundation style; every one draws this run's data)


def _tokens(ax: Any) -> dict[str, str]:
    return es.stage_tokens(es.axes_theme(ax))


def _ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _side_label_bar(fig: Any, mappable: Any, ax: Any, label: str) -> Any:
    """A slim colour bar whose unit runs along the bar's inner side. A title above a slim bar
    at the figure's right edge runs off the figure once the unit is longer than a word."""
    bar = es.colourbar_px(fig, mappable, ax, label="", shrink=0.8)
    bar.ax.yaxis.set_label_position("left")
    bar.set_label(label, fontsize=es.TICK_SIZE, color=_tokens(ax)["text"], labelpad=2)
    # round 9: a log bar gets a number at each decade inside its range (not only the top one)
    from matplotlib.colors import LogNorm
    norm = getattr(mappable, "norm", None)
    if isinstance(norm, LogNorm) and norm.vmin and norm.vmax:
        decades = [v for v in (1, 10, 100, 1000, 10000) if norm.vmin <= v <= norm.vmax]
        if len(decades) >= 2:
            bar.set_ticks(decades)
            bar.set_ticklabels([f"{v:g}" for v in decades])
    return bar


#: Shortest interval between two redraws of the solver's 2D panels while playing (ms).
CURSOR_INTERVAL_MS = 50


def _step_label(nits: list[int], where: float, *, short: bool = False) -> str:
    """Where a playback position sits (ED3-8): "the start", "step 4" or, between two
    recorded steps, "between the start and step 1" / "between step 1 and step 2"
    (``short``: "between start and step 1", for narrow panel subtitles)."""
    k = int(np.floor(where + 1e-9))
    start = "start" if short else "the start"
    if where < 1e-9:
        return start if short else "the start"
    if abs(where - k) < 1e-9 or k + 1 >= len(nits):
        return f"step {nits[min(k, len(nits) - 1)]}"
    left = start if k == 0 else f"step {nits[k]}"
    joiner = "&" if short else "and"  # narrow panels: the title must not elide
    return f"between {left} {joiner} step {nits[k + 1]}"


def _at_step(nits: list[int], where: float) -> str:
    """"at step 4" / "at the start", but "between …" needs no "at" in front of it."""
    label = _step_label(nits, where)
    return label if label.startswith("between") else f"at {label}"


def interpolated_camera_means(states: Any, position: float) -> np.ndarray:
    """Each camera's mean error at a fractional playback position, interpolated on the log
    scale between the two recorded states (as C's cursor, P3-1)."""
    values = np.array([s.camera_mean_px for s in states], dtype=float)
    position = float(np.clip(position, 0, len(values) - 1))
    lo = int(np.floor(position + 1e-9))
    hi = min(lo + 1, len(values) - 1)
    s = position - lo if hi != lo else 0.0
    a, b = np.log(np.maximum(values[lo], 1e-12)), np.log(np.maximum(values[hi], 1e-12))
    out = np.exp((1 - s) * a + s * b)
    return np.where(np.isfinite(values[lo]) & np.isfinite(values[hi]), out, np.nan)


#: Shared-photo line widths in the pose graph (UX-36 / ED3-7): linear from the fewest to the
#: most photos any linked pair shared, so pairs can be told apart even when every pair is
#: linked; the thinnest stays thin enough that 28 lines remain a graph, not a blot.
GRAPH_LINEWIDTHS = (0.6, 6.0)


def draw_pose_graph(ax: Any, n_cams: int, positions: np.ndarray, shared: np.ndarray | None,
                    events: list[dict], upto: int, level: int, anchor: int = 0) -> None:
    """Pose graph seen from above: badges where the cameras stand, grey lines for photos two
    cameras shared (widths from the fewest to the most shared, with a width key), accent
    arrows for the first guess's joins, each with a square ``#k`` tag 35 % along it (UX-5 /
    UX2-10: never a circle, which is a camera badge; tags kept clear of badges and of each
    other, at every level)."""
    from matplotlib.patches import FancyArrowPatch
    t = _tokens(ax)
    ax.set_aspect("equal", adjustable="datalim")  # fill the cell: titles line up
    ax.set_axis_off()
    pos = np.asarray(positions, dtype=float)
    if shared is not None and n_cams > 1:
        off = shared.astype(float).copy()
        np.fill_diagonal(off, 0)
        values = off[np.triu_indices(n_cams, 1)]
        values = values[values > 0]
        # ED3-7: when every pair shares photos (a complete graph), widths scaled from zero make
        # 51 and 67 photos look alike. Scale from the fewest to the most any linked pair
        # shared, so the width tells the pairs apart.
        low = float(values.min()) if values.size else 0.0
        top = float(values.max()) if values.size else 1.0

        def width_of(count: Any) -> np.ndarray:
            count = np.asarray(count, dtype=float)
            if top - low < 1e-9:
                return np.full(count.shape, 0.5 * (GRAPH_LINEWIDTHS[0] + GRAPH_LINEWIDTHS[1]))
            share = np.clip((count - low) / (top - low), 0, 1)
            return GRAPH_LINEWIDTHS[0] + (GRAPH_LINEWIDTHS[1] - GRAPH_LINEWIDTHS[0]) * share

        widths = width_of(off)
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                if off[i, j] > 0:
                    ax.plot(*pos[[i, j]].T, color=t["muted"], alpha=0.30,
                            linewidth=float(widths[i, j]), solid_capstyle="round", zorder=1)
        if values.size:
            keys = sorted({int(values.min()), int(values.max())})
            handles = es.weight_legend_handles(keys, t["muted"], vmax=top)
            for handle, key in zip(handles, keys):
                handle.set_linewidth(float(width_of([key])[0]))
                handle.set_alpha(0.55)
            ax.legend(handles=handles, loc="best", fontsize=8, frameon=False,
                      handlelength=1.6, borderaxespad=0.0, title="shared photos",
                      title_fontsize=8)
    joined = {int(anchor)}
    shown = events[:max(0, int(upto))]
    centre = pos.mean(axis=0)
    span = float(np.ptp(pos, axis=0).max()) or 1.0
    # Tag and badge sizes in data units, from the axes' size on screen (equal aspect): the
    # tags are kept apart as rectangles of their drawn size, not by a fraction of the graph.
    box = ax.get_window_extent()
    pad = 0.34
    per_px = max((np.ptp(pos[:, 0]) + 2 * pad) / max(box.width, 1.0),
                 (np.ptp(pos[:, 1]) + 2 * pad) / max(box.height, 1.0)) / 0.85
    size = (26.0 * per_px, 19.0 * per_px, 21.0 * per_px)  # tag width, height, badge radius
    placed_tags: list[np.ndarray] = []
    for order, event in enumerate(shown, start=1):
        a, b = int(event.get("joined_from", anchor)), int(event.get("joining", anchor))
        if not (0 <= a < n_cams and 0 <= b < n_cams):
            continue
        current = order == len(shown)
        ax.add_patch(FancyArrowPatch(
            tuple(pos[a]), tuple(pos[b]), arrowstyle="-|>,head_length=0.55,head_width=0.32",
            mutation_scale=14, color=t["accent"], linewidth=3.0 if current else 2.0,
            shrinkA=16, shrinkB=18, zorder=2, alpha=1.0 if current else 0.75))
        spot = _tag_spot(pos, a, b, centre, span, placed_tags, size)
        placed_tags.append(spot)
        ax.text(*spot, f"#{order}", ha="center", va="center", fontsize=9 if level < 3 else 8,
                fontweight="bold", color=t["accent"], zorder=4,
                bbox=dict(boxstyle="square,pad=0.18", facecolor=t["stage"],
                          edgecolor=t["accent"], linewidth=0.9))
        joined.update((a, b))
    current_cam = int(shown[-1]["joining"]) if shown else None
    on = [c for c in range(n_cams) if c in joined]
    off_cams = [c for c in range(n_cams) if c not in joined]
    if on:
        es.draw_camera_badges(ax, pos[on], on, size=14,
                              highlight=[current_cam] if current_cam is not None else [],
                              zorder=6)
    for c in off_cams:  # not joined yet: a muted, dashed badge
        ax.text(*pos[c], es.camera_badge_label(c), ha="center", va="center", fontsize=14,
                fontweight="bold", color=t["muted"], zorder=5, alpha=0.8,
                bbox=dict(boxstyle="circle,pad=0.39", facecolor=t["stage"],
                          edgecolor=t["muted"], linewidth=1.2, linestyle=(0, (2, 2))))
    # Invisible corners set the data limits; the equal aspect then widens them to fill the cell.
    pad = 0.34
    ax.scatter([pos[:, 0].min() - pad, pos[:, 0].max() + pad],
               [pos[:, 1].min() - pad, pos[:, 1].max() + pad], s=0, alpha=0)
    ax.margins(0)


def _tag_spot(pos: np.ndarray, a: int, b: int, centre: np.ndarray, span: float,
              taken: list[np.ndarray], size: tuple[float, float, float] | None = None
              ) -> np.ndarray:
    """Where a join tag goes: beside its arrow (perpendicular, away from the middle of the
    graph), 35 % along it from the camera it starts at, else at the first of a few other spots
    that keeps clear of the badges and of the tags already placed (UX-5, UX2-10)."""
    d = pos[b] - pos[a]
    length = float(np.linalg.norm(d)) or 1.0
    normal = np.array([-d[1], d[0]]) / length
    w, h, r = size if size is not None else (0.12 * span, 0.08 * span, 0.15 * span)
    # Round 10 (UX5-5): half the tag box's extent across the join line; a tag must clear the
    # line by a margin, or it sits on the arrow (the old fallback could land there)
    half_extent = abs(normal[0]) * w / 2 + abs(normal[1]) * h / 2
    best, best_gap = None, -np.inf
    for offset in (0.9, 1.3, 1.8, 2.4, 3.0):
        for along in (0.35, 0.45, 0.28, 0.55, 0.62, 0.22, 0.7, 0.8):
            mid = pos[a] + along * d
            for sign in (1.0, -1.0):
                n = normal if normal @ (mid - centre) >= 0 else -normal
                spot = mid + sign * offset * max(w, h) * n
                badge_gap = min(np.linalg.norm(spot - p) for p in pos) / (r + 0.6 * w)
                tag_gap = (min(max(abs(spot[0] - q[0]) / (1.1 * w), abs(spot[1] - q[1]) / (1.1 * h))
                               for q in taken)) if taken else np.inf
                line_clear = (abs((spot - pos[a]) @ normal) - half_extent) / (0.35 * max(w, h))
                gap = min(badge_gap, tag_gap, line_clear)
                if gap >= 1.0:
                    return spot
                if gap > best_gap:
                    best, best_gap = spot, gap
    return best if best is not None else pos[a] + 0.5 * d


def draw_join_order(ax: Any, events: list[dict], upto: int, anchor: int,
                    n_cams: int, badge_size: float = 11, *, candidates: Any = None,
                    unit_scale: float | None = None) -> None:
    """The first guess's joins in order (ED2-11), each with the score that chose it: the
    disagreement of the link between the two cameras (how far the photos both placed on
    their own disagree about where one camera sits relative to the other; log axis, mm when
    ``unit_scale`` converts target units to mm). With ``candidates``
    (:class:`education_visual_p3.JoinCandidates`, reproduced), every link open at that moment
    is a grey dot and the one taken, always the smallest, is the accent dot: the criterion is
    drawn, not just named."""
    from matplotlib.transforms import blended_transform_factory
    t = _tokens(ax)
    ax.set_axis_off()
    rows = len(events) + 1
    ax.set_ylim(rows - 0.35, -1.05)
    ax.set_xlim(0, 1)
    scale = float(unit_scale) if unit_scale else 1.0
    chosen = [float(e["spread"]) * scale if e.get("spread") is not None else np.nan
              for e in events]
    options = ([[v * scale for v in opts] for opts in candidates.options]
               if candidates is not None and candidates.reproduced else None)
    values = [v for v in chosen if np.isfinite(v) and v > 0] + (
        [v for opts in options for v in opts if v > 0] if options else [])
    x0, x1 = 0.50, 0.97
    if values:
        lo_e = np.floor(np.log10(min(values)) * 2) / 2
        hi_e = np.ceil(np.log10(max(values)) * 2) / 2
        if hi_e - lo_e < 1:
            hi_e = lo_e + 1

        def xpos(v: float) -> float:
            return x0 + (x1 - x0) * (np.log10(v) - lo_e) / (hi_e - lo_e)

        for e in np.arange(np.ceil(lo_e), np.floor(hi_e) + 1e-9):
            for m in (1.0, 3.0):
                v = m * 10 ** e
                if 10 ** lo_e <= v <= 10 ** hi_e * 1.001:
                    ax.plot([xpos(v)] * 2, [0.45, rows - 0.55], color=t["grid"],
                            linewidth=0.8, zorder=1)
                    ax.text(xpos(v), rows - 0.45, f"{v:g}", ha="center", va="top",
                            fontsize=8, color=t["muted"])
    else:
        def xpos(v: float) -> float:
            return x0

    def badge(x, y, cam, alpha):
        if 0 <= cam < n_cams:
            ax.text(x, y, es.camera_badge_label(cam), ha="center", va="center",
                    fontsize=badge_size, fontweight="bold", color=es.badge_text_colour(cam, ax),
                    alpha=alpha, zorder=5,
                    bbox=dict(boxstyle="circle,pad=0.3", alpha=alpha,
                              **es.camera_mark_kwargs(cam, ax, "badge")))

    badge(0.12, 0, anchor, 1.0)
    ax.text(0.20, 0, "starts alone", ha="left", va="center", fontsize=10, color=t["text"])
    for k, event in enumerate(events, start=1):
        shown = k <= upto
        alpha = 1.0 if shown else es.REJECTED_ALPHA
        colour = t["accent"] if shown else t["muted"]
        ax.text(0.0, k, _ordinal(k), ha="left", va="center", fontsize=10, fontweight="bold",
                color=colour, alpha=alpha)
        badge(0.12, k, int(event.get("joining", -1)), alpha)
        ax.text(0.215, k, "via", ha="center", va="center", fontsize=10, color=t["muted"],
                alpha=alpha)
        badge(0.31, k, int(event.get("joined_from", -1)), alpha)
        if options is not None and k - 1 < len(options):
            others = [v for v in options[k - 1] if v > 0]
            ax.scatter([xpos(v) for v in others], np.full(len(others), k), s=16,
                       color=t["muted"], alpha=0.55 * alpha, zorder=2, linewidths=0)
        v = chosen[k - 1]
        if np.isfinite(v) and v > 0:
            ax.scatter([xpos(v)], [k], s=60, color=colour, alpha=alpha, zorder=3,
                       edgecolors=t["stage"], linewidths=0.8)
    where = blended_transform_factory(ax.transAxes, ax.transData)
    unit = "mm" if unit_scale else "target units"
    # ED3-7: say WHAT disagrees: where the shared photos put the joining camera.
    ax.text(x0, -0.72, f"spread of the camera's place over shared photos ({unit}, log)",
            transform=where, ha="left", va="center", fontsize=9, color=t["muted"])


def join_cmap(theme: Any) -> Any:
    """One-hue sequential map for counts (UX2-11): the shared amount map
    (:func:`education_style.amount_cmap`, UX3-3; moved there from this module)."""
    import matplotlib
    return matplotlib.colormaps[es.amount_cmap(theme)]


def draw_shared_matrix(ax: Any, shared: np.ndarray) -> Any:
    """Photos both cameras saw in the solve (diagonal: photos each camera saw), on a one-hue
    scale from 0 (UX2-11: 51 and 80 both read as 'many'), cells at most ~44 px, centred."""
    import matplotlib
    n = len(shared)
    cmap = join_cmap(ax)
    top = max(float(shared.max()), 1.0)
    image = ax.imshow(shared, cmap=cmap, vmin=0, vmax=top, aspect="equal",
                      interpolation="nearest")
    if n <= 12:
        for i in range(n):
            for j in range(n):
                value = int(shared[i, j])
                fill = matplotlib.colors.to_hex(cmap(value / top))
                colour = es.WHITE if es.relative_luminance(fill) < 0.35 else "#000000"
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color=colour)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if n < 6:  # UX2-11: a few cameras keep small cells, centred, not slabs filling the panel
        pad = (6 - n) / 2
        ax.set_xlim(-0.5 - pad, n - 0.5 + pad)
        ax.set_ylim(n - 0.5 + pad, -0.5 - pad)
    es._badge_ticks(ax, list(range(n)), list(range(n)), xlabel=None, size=9)
    es.badge_ticks_y(ax, list(range(n)), list(range(n)), size=10)
    return image
def draw_coverage_raster(ax: Any, counts: np.ndarray, excluded: list[int]) -> None:
    """Corners each camera saw in each photo: the camera's colour, stronger for more corners
    (the same encoding as Phase 4, UX-22); blank where it saw nothing; ✕ photos left out."""
    from matplotlib.colors import to_rgba
    t = _tokens(ax)
    n_cam, n_img = counts.shape
    image = np.zeros((n_cam, n_img, 4))
    top = max(float(counts.max()), 1.0)
    for cam in range(n_cam):
        rgba = np.array(to_rgba(es.camera_colour(cam, ax)))
        image[cam, :, :3] = rgba[:3]
        image[cam, :, 3] = np.where(counts[cam] > 0, 0.25 + 0.75 * counts[cam] / top, 0.0)
    ax.set_facecolor(t["stage"])
    ax.imshow(image, aspect="auto", interpolation="nearest")
    if excluded:
        ax.plot(excluded, np.full(len(excluded), -0.9), linestyle="none", marker="x",
                color=t["bad"], markersize=5, clip_on=False)
    ax.set_xlabel("Photo")
    ticks = [k for k in range(0, n_img, 10)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([es.photo_label(k, short=True) for k in ticks])
    es.badge_ticks_y(ax, list(range(n_cam)), list(range(n_cam)), size=7)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_tilt_ambiguity(ax_one: Any, ax_two: Any) -> None:
    """Side view, drawn from geometry: one face at +θ and -θ under parallel (telecentric) rays.

    Left: rays along one camera's axis project both tilts onto the same sensor interval.
    Right: a second camera at another angle sees two different intervals.
    """
    t = _tokens(ax_one)
    half, theta = 1.2, np.radians(35.0)
    # (tilt angle, label, colour, line style): style and label carry the meaning, not colour alone.
    tilts = ((-theta, "tilted towards the camera", t["accent"], "-"),
             (theta, "tilted away", t["warn"], "--"))
    centre = np.array([4.0, 0.0])
    for angle, label, colour, style in tilts:
        top = centre + half * np.array([np.sin(angle), np.cos(angle)])
        bottom = centre - half * np.array([np.sin(angle), np.cos(angle)])
        ax_one.plot(*np.stack([bottom, top]).T, style, color=colour, linewidth=2.5, label=label)
        for end in (top, bottom):
            ax_one.plot([0.0, end[0]], [end[1], end[1]], ":", color=t["muted"], linewidth=1)
    extent = half * np.cos(theta)
    ax_one.plot([0.0, 0.0], [-1.6, 1.6], color=t["muted"], linewidth=1)
    ax_one.plot([0.0, 0.0], [-extent, extent], color=t["text"], linewidth=4, solid_capstyle="butt")
    ax_one.text(0.0, -1.75, "sensor", ha="center", va="top", fontsize=9, color=t["muted"])
    view = np.radians(40.0)  # not 55°: that views the 35° 'away' tilt edge-on
    direction = np.array([np.cos(view), np.sin(view)])
    sensor_axis = np.array([-np.sin(view), np.cos(view)])
    for offset, (angle, label, colour, style) in zip((4.0, 4.4), tilts):
        sensor = -offset * direction
        ends = [sign * half * np.array([np.sin(angle), np.cos(angle)]) for sign in (1, -1)]
        images = [sensor + (end @ sensor_axis) * sensor_axis for end in ends]
        ax_two.plot(*np.stack(ends).T, style, color=colour, linewidth=2.5, label=label)
        for end, image in zip(ends, images):
            ax_two.plot([image[0], end[0]], [image[1], end[1]], ":", color=t["muted"],
                        linewidth=1)
        ax_two.plot(*np.stack(images).T, style, color=colour, linewidth=4, solid_capstyle="butt")
    for axes in (ax_one, ax_two):
        axes.set_aspect("equal", adjustable="datalim")
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)


def draw_candidate_fits(ax: Any, image: Any, candidates: list[Mapping], level: int,
                        camera_index: Mapping[str, int]) -> None:
    """Per camera that saw the photo: its two candidate tilts' fits (log px, 20 px limit)."""
    t = _tokens(ax)
    if not candidates:
        es.empty_panel(ax, "No recovered photo in this run" if image is None else
                       f"Candidate fits for {es.photo_label(int(image))} were not recorded")
        return

    def value(entry: Mapping, key: str) -> float:
        try:
            number = float(entry.get(key))
        except (TypeError, ValueError):
            return np.nan
        return max(number, 0.01) if np.isfinite(number) else np.nan

    series = [("own_camera_rms_px", "fit in its own camera", 0.6, "///"),
              ("all_cameras_tilt_kept_rms_px", "all cameras, tilt kept", 1.0, None)]
    if level >= 2:
        series.append(("all_cameras_rms_px", "all cameras, refined freely", 0.0, None))
    series = [s for s in series if any(s[0] in entry for entry in candidates)]
    cameras: list[str] = []
    for entry in candidates:
        name = str(entry.get("camera"))
        if name not in cameras:
            cameras.append(name)
    width = 0.8 / (2 * max(len(series), 1))
    from matplotlib.patches import Patch
    kept_fits: list[float] = []
    for c_pos, name in enumerate(cameras):
        index = camera_index.get(name)
        colour = es.camera_colour(index, ax) if index is not None else t["muted"]
        edge = es.camera_label_colour(ax, index) if index is not None else t["muted"]
        for t_pos, tilt in enumerate(("+", "-")):
            entry = next((e for e in candidates if str(e.get("camera")) == name
                          and str(e.get("tilt")) in ((tilt,) if tilt == "+" else ("-", "−"))),
                         None)
            if entry is None:
                continue
            for s_pos, (key, _label, alpha, hatch) in enumerate(series):
                x = c_pos + (t_pos - 0.5) * 0.42 + (s_pos - (len(series) - 1) / 2) * width
                height = value(entry, key)
                if alpha == 0.0:
                    ax.bar(x, height, width * 0.92, facecolor="none", edgecolor=edge,
                           linewidth=1.3, linestyle=(0, (2, 1.5)), zorder=3)
                else:
                    ax.bar(x, height, width * 0.92, color=colour, alpha=alpha, hatch=hatch,
                           edgecolor=edge, linewidth=0.6, zorder=3)
            ax.annotate("+" if tilt == "+" else "−", (c_pos + (t_pos - 0.5) * 0.42, 0),
                        xycoords=("data", "axes fraction"), xytext=(0, -2),
                        textcoords="offset points", ha="center", va="top", fontsize=10,
                        fontweight="bold", color=t["text"])
        # ED2-10: mark the decision, the tilt that fits every camera with its turn kept.
        kept = [(value(e, "all_cameras_tilt_kept_rms_px"), str(e.get("tilt")))
                for e in candidates if str(e.get("camera")) == name]
        kept = [k for k in kept if np.isfinite(k[0])]
        if kept:
            best_value, best_tilt = min(kept)
            kept_fits.append(best_value)
            t_pos = 0 if best_tilt == "+" else 1
            ax.annotate("✓", (c_pos + (t_pos - 0.5) * 0.42, best_value), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=12,
                        fontweight="bold", color=t["good"])
    # UX3-11: neutral unless a kept fit is above the limit (the mirror tilt is meant to miss).
    style = es.limit_style(ax, es.limit_crossed(kept_fits, 20.0))
    ax.axhline(20.0, **style)
    ax.annotate("20 px limit", (1.0, 20.0), xycoords=("axes fraction", "data"), xytext=(4, 0),
                textcoords="offset points", ha="left", va="center", fontsize=9,
                color=style["color"], annotation_clip=False)
    ax.set_yscale("log")
    _plain_log_axis(ax)
    ax.set_xlim(-0.6, len(cameras) - 0.4)
    low, high = ax.get_ylim()
    # Headroom for the top-left legend (three rows) so the ✓ marks sit clear of it.
    ax.set_ylim(min(low, 0.05), max(high, 60) * 16.0)
    ax.set_ylabel(es.error_label("rms", short=True))
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    indices = [camera_index.get(name) for name in cameras]
    if all(i is not None for i in indices):
        es._badge_ticks(ax, list(range(len(cameras))), indices, xlabel=None, size=10)
        ax.tick_params(axis="x", pad=24)
    else:
        ax.set_xticks(range(len(cameras)))
        ax.set_xticklabels(cameras, fontsize=9)
    handles = [Patch(facecolor=t["muted"], alpha=a if a else 1.0, hatch=h,
                     edgecolor=t["muted"], fill=a != 0.0, linestyle="-" if a else "--",
                     label=label) for _key, label, a, h in series]
    # One entry per row: three entries in a row ran off the panel's right edge at 1280 px.
    # Tighter handles and padding keep the box clear of the ✓ marks at the right of the panel.
    ax.legend(handles=handles, loc="upper left", ncols=1, fontsize=8.5, handlelength=1.4,
              handletextpad=0.5, borderpad=0.3, bbox_to_anchor=(0.0, 1.0), frameon=True,
              facecolor=t["stage"], framealpha=0.92, edgecolor="none")


def draw_tilt_demo(ax: Any, photo: np.ndarray | None, demo: Any, size: tuple[int, int],
                   *, mirror_label: str = "wrong-side tilt ✕ (rejected)", final_px: float | None = None) -> dict:
    """ED2-10 (L1): one recovered photo in a second camera: the corners it found (rings), where
    the kept tilt puts them (dots, ✓) and where the mirror tilt would (✕). Both tilts fit the
    first camera equally; only one lands on the second camera's rings. ``final_px``: the saved
    fit once the tilt is freed too (ED3-6: the picker's number, bridged in the legend)."""
    t = _tokens(ax)
    if photo is not None:
        ax.imshow(photo, cmap="gray", interpolation="antialiased", vmin=0, vmax=255, zorder=0)
        size = (photo.shape[1], photo.shape[0])
    else:
        ax.set_facecolor(t["stage"])
    found = np.asarray(demo.found, dtype=float)
    kept = np.asarray(demo.predicted[demo.kept], dtype=float)
    other_tilt = "-" if demo.kept == "+" else "+"
    other = np.asarray(demo.predicted[other_tilt], dtype=float)
    from pyCamSet.workflow.education_visual_p3 import photo_crop
    w, h = size if all(size) else (int(np.nanmax(found[:, 0]) + 40), int(np.nanmax(found[:, 1]) + 40))
    crop = photo_crop(found, [kept, other], (w, h), margin=0.12)
    span = max(abs(crop[1] - crop[0]), abs(crop[2] - crop[3]), 1.0)
    from matplotlib.collections import EllipseCollection, LineCollection
    ring = max(3.0, 0.016 * span)
    ax.add_collection(EllipseCollection(
        np.full(len(found), 2 * ring), np.full(len(found), 2 * ring), np.zeros(len(found)),
        units="xy", offsets=found, offset_transform=ax.transData, facecolors="none",
        edgecolors=t["accent"], linewidths=1.5, zorder=3))
    ax.add_collection(LineCollection(np.stack([other, found], axis=1), colors=t["bad"],
                                     linewidths=0.8, alpha=0.55, zorder=2))
    ax.scatter(other[:, 0], other[:, 1], marker="x", s=30, color=t["bad"], linewidths=1.6,
               zorder=4, label=f"{mirror_label}: {es.format_px(demo.rms[other_tilt])}")
    ax.scatter(kept[:, 0], kept[:, 1], s=26, color=t["good"], zorder=5,
               edgecolors=t["stage"], linewidths=0.6,
               label=f"kept tilt ✓: {es.format_px(demo.rms[demo.kept])} (tilt not refitted)"
               + (f" → {es.format_px(final_px)} with the tilt also fitted" if final_px is not None
                  and np.isfinite(final_px) else ""))
    ax.set_xlim(crop[0], crop[1])
    ax.set_ylim(crop[2], crop[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    es.frame_axes(ax, demo.viewer)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=2, fontsize=9,
              frameon=False, borderaxespad=0.3, handletextpad=0.3, columnspacing=1.2)
    return {"crop": crop}


def _objective_header(playback: Mapping | None) -> str:
    """SCI6-1: the playback column is the solver's score. ½ Σ squared error (px²) only for a
    linear loss; a robust loss (for example soft_l1) is a different sum, so name the loss."""
    loss = str(((playback or {}).get("payload") or {}).get("loss") or "linear")
    if loss == "linear":
        return "Solver score: ½ Σ squared error (px²)"
    return f"Solver score: {loss} robust loss (not ½ Σ px²)"


def _reason_tag(reason: str) -> str:
    """A two-word tag for a left-out photo's saved reason (the full text is in the table)."""
    reason = str(reason or "")
    if reason.startswith("seen by one camera only"):
        return "1 camera"
    if "exceeds the 20 px limit" in reason:
        return "> 20 px"
    if reason.startswith("recovery off"):
        return "off"
    return "other"


def draw_recovery_overview(ax: Any, recovered: list[Mapping], left_out: list[Mapping],
                           selected: Any, level: int = 2) -> None:
    """Every single-face photo: placed (fit error) or left out, with the 20 px limit. A left-out
    photo's marker says why: ✕ seen by one camera only, ▲ its best fit was above 20 px."""
    t = _tokens(ax)
    rows = [(int(e.get("image", -1)), float(e.get("rms_px", np.nan)), True,
             len(e.get("cameras") or []), "") for e in recovered]
    rows += [(int(e.get("image", -1)), np.nan, False, len(e.get("cameras") or []),
              _reason_tag(e.get("reason"))) for e in left_out]
    rows.sort()
    if not rows:
        es.empty_panel(ax, "No single-face photo in this run")
        return
    x = np.arange(len(rows))
    placed = [(i, v) for i, (_img, v, ok, _n, _r) in enumerate(rows) if ok]
    if placed:
        xs, ys = zip(*placed)
        ax.scatter(xs, ys, s=42, color=t["accent"], zorder=3, label="placed")
    top = max([v for _i, v in placed if np.isfinite(v)] + [20.0]) * 1.4
    markers = {"1 camera": ("x", "left out: 1 camera"), "> 20 px": ("^", "left out: > 20 px"),
               "off": ("s", "left out: recovery off"), "other": ("D", "left out: other")}
    for tag, (marker, label) in markers.items():
        lost = [i for i, (_img, _v, ok, _n, r) in enumerate(rows) if not ok and r == tag]
        if lost:
            ax.scatter(lost, np.full(len(lost), top * 0.84), marker=marker, s=46,
                       color=t["bad"], zorder=3, label=label)
    for i, (img, _value, _ok, _n_cams, _reason) in enumerate(rows):
        if str(img) == str(selected):
            # UX5-3: a column that stops short of the top, not an axvline (which ran the full
            # height into the y-axis top and the legend row)
            ax.vlines(i, 0, top * 1.12, color=t["text"], alpha=0.25, linewidth=8, zorder=1,
                      label="selected photo")
    es.limit_line(ax, 20.0, values=[v for _i, v in placed])
    ax.set_ylim(0, top * 1.32)
    ax.set_xticks(x)
    ax.set_xticklabels([es.photo_label(row[0], short=True) for row in rows], fontsize=9,
                       rotation=90 if len(rows) > 14 else 0)
    ax.set_xlabel("Photo")
    ax.set_ylabel(es.error_label("rms", short=True))
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.legend(loc="upper center", ncols=3, fontsize=8, frameon=False, borderaxespad=0.2,
              handletextpad=0.2, columnspacing=0.8)


def _plain_value_axis(ax: Any) -> None:
    """No offset / scientific notation on a value axis (a "1e-12" label would sit on the
    subtitle); an all-zero axis gets a 0..1 range."""
    formatter = ax.yaxis.get_major_formatter()
    if hasattr(formatter, "set_useOffset"):
        formatter.set_useOffset(False)
        formatter.set_scientific(False)
    low, high = ax.get_ylim()
    if max(abs(low), abs(high)) < 1e-9:
        ax.set_ylim(0, 1)


def _plain_log_axis(ax: Any) -> None:
    """Label a log y axis with plain numbers (3, 10, 100), not powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _curve_point(x: np.ndarray, y: np.ndarray, position: float) -> tuple[float, float]:
    lo = int(np.clip(np.floor(position), 0, len(x) - 1))
    hi = min(lo + 1, len(x) - 1)
    s = float(position - lo) if hi != lo else 0.0
    px = (1 - s) * x[lo] + s * x[hi]
    py = float(np.exp((1 - s) * np.log(max(y[lo], 1e-12)) + s * np.log(max(y[hi], 1e-12))))
    return float(px), py


def draw_error_curve(ax: Any, nits: list[int], means: list[float], position: float | None,
                     *, inset: bool = True) -> dict:
    """Mean error (px, log) per recorded step; a cursor marks the shown position. When the
    first step does almost all the work, an inset zooms on the later steps (UX-11)."""
    t = _tokens(ax)
    x = np.asarray(nits, dtype=float)
    y = np.asarray(means, dtype=float)
    ax.plot(x, y, color=t["accent"], linewidth=2.0, zorder=2)
    ax.scatter(x, y, s=22, color=t["accent"], zorder=3)
    ax.set_yscale("log")
    _plain_log_axis(ax)
    finite = y[np.isfinite(y) & (y > 0)]
    if finite.size:
        ax.set_ylim(finite.min() / 1.6, finite.max() * 1.9)
    ax.set_xlabel("Solver step")
    ax.set_ylabel(es.error_label(short=True))
    ax.yaxis.grid(True, which="major", color=t["grid"], linewidth=0.8)
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    es.integer_axis(ax, "x")
    artists: dict[str, Any] = {"x": x, "y": y}
    if position is not None:
        px, py = _curve_point(x, y, position)
        artists["line"] = ax.axvline(px, color=t["text"], alpha=0.45, linewidth=1.2, zorder=1)
        artists["dot"] = ax.scatter([px], [py], s=90, facecolors=t["stage"],
                                    edgecolors=t["text"], linewidths=2.0, zorder=4)
    later = y[1:]
    if inset and len(later) >= 3 and np.isfinite(later).all() and y[0] > 3 * later.max():
        sub = ax.inset_axes([0.40, 0.40, 0.57, 0.50])
        sub.set_zorder(6)  # above the main cursor line
        # round 9: an opaque inset, so the cursor line does not run through its title
        sub.set_facecolor(ax.get_facecolor())
        sub.plot(x[1:], later, color=t["accent"], linewidth=1.6)
        sub.scatter(x[1:], later, s=12, color=t["accent"])
        spread = max(float(later.max() - later.min()), 1e-6)
        sub.set_ylim(later.min() - 0.15 * spread, later.max() + 0.35 * spread)
        sub.set_xlim(x[1] - 0.5, x[-1] + 0.5)
        # UX8-4: the inset is the only scale of the fine-tuning steps; same size as the main ticks
        # round 10 (UX8-4): one size step up, the inset's ticks were the smallest on the page
        sub.tick_params(labelsize=es.LABEL_SIZE)
        sub.set_facecolor(t["stage"])
        sub.yaxis.grid(True, color=t["grid"], linewidth=0.6)
        from matplotlib.ticker import MaxNLocator
        sub.yaxis.set_major_locator(MaxNLocator(3))
        sub.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        sub.set_title(f"steps {int(x[1])}–{int(x[-1])}: fine tuning", fontsize=9,
                      color=t["muted"], loc="left", pad=3)
        for spine in sub.spines.values():
            spine.set_color(t["grid"])
        artists["inset"] = sub
        if position is not None:
            px, py = _curve_point(x, y, max(position, 1.0))
            artists["inset_dot"] = sub.scatter([px], [py], s=40, facecolors=t["stage"],
                                               edgecolors=t["text"], linewidths=1.6, zorder=4)
            artists["inset_dot"].set_visible(position >= 1.0)
    return artists


def draw_camera_error_bars(ax: Any, values: np.ndarray, limits: tuple[float, float],
                           start: np.ndarray | None, highlight: int | None) -> list:
    """Per-camera mean error at the shown step (log px); the start as grey ticks."""
    t = _tokens(ax)
    cams = list(range(len(values)))
    safe = np.where(np.isfinite(values), values, np.nan)
    bars = ax.bar(cams, safe, width=0.62, zorder=3,
                  color=[es.camera_colour(c, ax) for c in cams], linewidth=0)
    if start is not None:
        ax.scatter(cams, start, marker="_", s=260, color=t["muted"], linewidths=2.0, zorder=4,
                   label="at the start")
    labels = [ax.annotate(es.format_number(v), (c, v), xytext=(0, 3), textcoords="offset points",
                          ha="center", va="bottom", fontsize=9, color=t["text"], gid="edu:value")
              for c, v in zip(cams, safe) if np.isfinite(v)]
    ax.set_yscale("log")
    _plain_log_axis(ax)
    ax.set_ylim(*limits)
    ax.set_xlim(-0.6, len(cams) - 0.4)
    ax.set_ylabel(es.error_label(short=True))
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    es._badge_ticks(ax, cams, cams, highlight=[highlight] if highlight is not None else [],
                    xlabel=None)
    return [bars, labels]


def draw_photo_panel(ax: Any, photo: np.ndarray | None, observed: np.ndarray,
                     predicted: np.ndarray, errors: np.ndarray, crop: tuple, *,
                     camera: int, size: tuple[int, int], scale: Any) -> dict:
    """One camera's photo: the detected corners as rings (theme accent) and where the camera
    model, at this state, predicts them, as dots coloured by error, joined by a line (ED-1).
    Returns the artists to update while playing."""
    from matplotlib.collections import LineCollection
    t = _tokens(ax)
    if photo is not None:
        ax.imshow(photo, cmap="gray", interpolation="antialiased", vmin=0, vmax=255, zorder=0)
    else:
        ax.set_facecolor(t["panel"] if "panel" in t else t["stage"])
    w, h = size
    x0, x1, y0, y1 = crop
    span = max(abs(x1 - x0), abs(y1 - y0), 1.0)
    ring = max(4.0, 0.018 * span)
    from matplotlib.collections import EllipseCollection
    rings = EllipseCollection(np.full(len(observed), 2 * ring), np.full(len(observed), 2 * ring),
                              np.zeros(len(observed)), units="xy", offsets=observed,
                              offset_transform=ax.transData, facecolors="none",
                              edgecolors=t["accent"], linewidths=1.6, zorder=3)
    ax.add_collection(rings)
    segments = np.stack([predicted, observed], axis=1)
    colours = scale.colours(errors)
    lines = LineCollection(segments, colors=colours, linewidths=1.3, alpha=0.9, zorder=4)
    ax.add_collection(lines)
    dots = ax.scatter(predicted[:, 0], predicted[:, 1], c=errors, s=26, zorder=5,
                      **scale.scatter_kwargs(), **{k: v for k, v in es.value_dot_kwargs(ax).items()
                                                   if k != "s"})
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    es.frame_axes(ax, camera)
    del w, h
    return {"dots": dots, "lines": lines}


def photo_magnification(span: float, start_error: float) -> int:
    """How much the dot-to-ring distances are enlarged so a few-pixel error is visible on a
    large photo (labelled "×N" in the subtitle, as Phase 2's arrows): 1 when the start error is
    already about 3 % of the shown width."""
    if not (np.isfinite(start_error) and start_error > 0):
        return 1
    wanted = 0.03 * span / start_error
    best = 1
    for f in (1, 2, 5, 10, 20):
        if f <= wanted:
            best = f
    return best


def update_photo_panel(artists: Mapping, observed: np.ndarray, predicted: np.ndarray,
                       errors: np.ndarray, scale: Any) -> None:
    """Move the predicted dots (and their lines) to another state."""
    factor = float(artists.get("magnify", 1))
    predicted = observed + factor * (np.asarray(predicted) - observed)
    artists["dots"].set_offsets(predicted)
    artists["dots"].set_array(np.asarray(errors, dtype=float))
    artists["lines"].set_segments(np.stack([predicted, observed], axis=1))
    artists["lines"].set_colors(scale.colours(errors))


def draw_normal_matrix(ax: Any, blocks: Mapping[str, Any]) -> None:
    """The normal matrix JᵀJ the solver builds, at full size: camera blocks (B) in camera
    colours, camera-photo couplings (E) muted, each photo's own 6x6 block (C) in the accent."""
    from matplotlib.colors import to_rgba
    t = _tokens(ax)
    codes = blocks["codes"]
    n_cam = int(blocks["n_cameras"])
    palette = np.zeros((n_cam + 3, 4))
    palette[0] = to_rgba(t["stage"])
    for c in range(n_cam):
        palette[c + 1] = to_rgba(es.camera_colour(c, ax))
    palette[n_cam + 1] = to_rgba(t["muted"], 0.55)
    palette[n_cam + 2] = to_rgba(t["accent"])
    ax.imshow(palette[codes.astype(int) + 1], interpolation="nearest", aspect="equal")
    keep = int(blocks["n_kept"])
    n = codes.shape[0]
    for v in (keep - 0.5,):
        ax.axhline(v, color=t["text"], linewidth=0.8, alpha=0.6)
        ax.axvline(v, color=t["text"], linewidth=0.8, alpha=0.6)
    ax.set_xticks([keep / 2, keep + (n - keep) / 2])
    ax.set_xticklabels([f"cameras\n{keep}", f"photo poses\n{n - keep}"], fontsize=9)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_color(t["grid"])


def reduced_system_blocks(system: Mapping[str, Any]) -> np.ndarray:
    """Size (Frobenius norm) of each camera-pair block of S, relative to the largest block.
    (The picture uses :func:`education_visual_p3.coupling_matrix`, which normalises out the
    mixed units; this raw form is kept for checks.)"""
    S = np.asarray(system["S"], dtype=float)
    slices = system["slices"]
    blocks = np.array([[np.linalg.norm(S[a0:a1, b0:b1]) for b0, b1 in slices]
                       for a0, a1 in slices])
    return blocks / max(float(blocks.max()), 1e-300)


def draw_coupling(ax: Any, coupling: np.ndarray) -> Any:
    """The reduced camera system: how strongly each pair of cameras is tied (0..1)."""
    t = _tokens(ax)
    n = len(coupling)
    # UX3-3: a tie is an amount, not an error: the one-hue amount map, never viridis.
    shown = ax.imshow(np.where(coupling > 0, coupling, np.nan), cmap=es.amount_cmap(ax),
                      vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    shown._edu_quantity = "amount"
    ax.set_facecolor(t["stage"])
    for k in range(1, n):
        ax.axhline(k - 0.5, color=t["stage"], linewidth=1.5)
        ax.axvline(k - 0.5, color=t["stage"], linewidth=1.5)
    cams = list(range(n))
    es._badge_ticks(ax, cams, cams, xlabel=None, size=8)
    es.badge_ticks_y(ax, cams, cams, size=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return shown


def draw_lens_grid(ax: Any, photo: np.ndarray | None, change: Any, cam: Any, factor: float,
                   camera: int) -> None:
    """The camera's photo with a straight grid (muted) and the same grid bent by the fitted
    distortion, exaggerated ``factor``× (accent); + marks the principal point Phase 2 used and
    the one Phase 3 fitted."""
    from pyCamSet.workflow.education_visual_p2 import distortion_grid
    t = _tokens(ax)
    size = change.image_size
    if photo is not None:
        ax.imshow(photo, cmap="gray", interpolation="antialiased", vmin=0, vmax=255, zorder=0,
                  alpha=0.75)
    straight, bent = distortion_grid(cam, size, factor, lines=9, samples=60,
                                     telecentric=change.telecentric)
    for line in straight:
        ax.plot(line[:, 0], line[:, 1], color=t["muted"], linewidth=0.8, alpha=0.7, zorder=2)
    if change.distortion_fitted:
        for line in bent:
            ax.plot(line[:, 0], line[:, 1], color=t["accent"], linewidth=1.5, zorder=3)
    before, after = change.centre_before, change.centre_after
    del before
    ax.plot(*after, marker="+", markersize=16, markeredgewidth=2.6, color=t["accent"], zorder=6)
    w, h = size
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    es.frame_axes(ax, camera)


def draw_lens_effect(ax: Any, photo: np.ndarray | None, observed: np.ndarray,
                     moved: np.ndarray, centre: tuple[float, float], size: tuple[int, int],
                     camera: int, field: tuple[np.ndarray, np.ndarray] | None = None) -> int:
    """The whole photo with the found corners (rings) and, from each, an arrow for how far
    the tuned lens bending moved its predicted position (×N so the largest is ~8 % of the
    frame); + marks the point the bending is centred on. Returns N."""
    t = _tokens(ax)
    if photo is not None:
        ax.imshow(photo, cmap="gray", interpolation="antialiased", vmin=0, vmax=255, zorder=0,
                  alpha=0.85)
        size = (photo.shape[1], photo.shape[0])
    else:
        ax.set_facecolor(t["stage"])
    w, h = size
    observed = np.asarray(observed, dtype=float)
    moved = np.asarray(moved, dtype=float)
    every = np.vstack([moved] + ([np.asarray(field[1], dtype=float)] if field is not None else []))
    largest = float(np.linalg.norm(every, axis=1).max()) if len(every) else 0.0
    factor = 1
    for f in (1, 2, 3, 5, 10, 20):
        if largest > 0 and f * largest <= 0.08 * max(w, h):
            factor = f
    if field is not None and len(field[0]):
        # The same camera's corners in its other photos, faint: the whole bending field.
        fxy, fmv = np.asarray(field[0], dtype=float), np.asarray(field[1], dtype=float)
        ax.quiver(fxy[:, 0], fxy[:, 1], factor * fmv[:, 0], factor * fmv[:, 1], angles="xy",
                  scale_units="xy", scale=1, color=t["muted"], alpha=0.55, width=0.004,
                  headwidth=3.5, headlength=4, zorder=3)
    if len(observed):
        ax.quiver(observed[:, 0], observed[:, 1], factor * moved[:, 0], factor * moved[:, 1],
                  angles="xy", scale_units="xy", scale=1, color=t["accent"], width=0.006,
                  headwidth=3.5, headlength=4, zorder=4)
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=4.5, foreground=t["stage"])]
    ax.plot(*centre, marker="+", markersize=18, markeredgewidth=2.6, color=t["text"],
            zorder=6, path_effects=halo)
    # ED3-1: name the + where it is drawn: the distortion is centred on the principal point,
    # which is also the image of the optical axis.
    right = centre[0] < 0.62 * w
    ax.annotate("bend centre", centre, xytext=(13 if right else -13, -11),
                textcoords="offset points", ha="left" if right else "right", va="bottom",
                fontsize=9, color=t["text"], zorder=6,
                path_effects=[pe.withStroke(linewidth=3.0, foreground=t["stage"])])
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    es.frame_axes(ax, camera)
    return factor


def draw_bending_effect(ax: Any, off: Mapping[int, float], on: Mapping[int, float],
                        highlight: int | None) -> None:
    """Each camera's mean error with the lens bending switched off (hollow) and as solved
    (dot), on the shared log px axis, joined by an arrow: the effect, in the learner's px."""
    t = _tokens(ax)
    cams = sorted(set(off) | set(on))
    scale = es.error_scale(ax)
    for k, cam in enumerate(cams):
        a, b = off.get(cam, np.nan), on.get(cam, np.nan)
        colour = es.camera_colour(cam, ax)
        if np.isfinite(a) and np.isfinite(b) and abs(a - b) > 1e-9:
            ax.annotate("", (k, b), xytext=(k, a), arrowprops=dict(
                arrowstyle="-|>", color=t["muted"], linewidth=1.1, shrinkA=5, shrinkB=5))
        ax.scatter([k], [a], s=64, facecolors="none", edgecolors=colour, linewidths=1.8,
                   zorder=3)
        ax.scatter([k], [b], s=64, zorder=4, **es.camera_mark_kwargs(cam, ax, "dot"))
    values = [v for v in list(off.values()) + list(on.values()) if np.isfinite(v) and v > 0]
    scale.log_axis(ax, "y")
    lo, hi = scale.log_range(values)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.6, len(cams) - 0.4)
    ax.set_ylabel(es.error_label(short=True))
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    es._badge_ticks(ax, list(range(len(cams))), cams,
                    highlight=[highlight] if highlight is not None else [], xlabel=None)


def draw_later_shift(ax: Any, before: Mapping[int, float], after: Mapping[int, float],
                     highlight: int | None) -> None:
    """Largest distortion shift per camera, Phase 3 (hollow) and Phase 4 (dot), log px."""
    t = _tokens(ax)
    cams = sorted(set(before) | set(after))
    for k, cam in enumerate(cams):
        a, b = before.get(cam, np.nan), after.get(cam, np.nan)
        colour = es.camera_colour(cam, ax)
        if np.isfinite(a) and np.isfinite(b):
            ax.plot([k, k], [max(a, 1e-3), max(b, 1e-3)], color=t["grid"], linewidth=2.0,
                    zorder=1)
        if np.isfinite(a):
            ax.scatter([k], [max(a, 1e-3)], s=60, facecolors="none", edgecolors=colour,
                       linewidths=1.8, zorder=3)
        if np.isfinite(b):
            ax.scatter([k], [max(b, 1e-3)], s=60, zorder=4,
                       **es.camera_mark_kwargs(cam, ax, "dot"))
    values = [v for v in list(before.values()) + list(after.values()) if np.isfinite(v) and v > 0]
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    if values:
        lo = 10 ** np.floor(np.log10(min(values)))
        hi = 10 ** np.ceil(np.log10(max(values) * 1.05))
        ax.set_ylim(lo, hi)
        ticks = [m * 10.0 ** e for e in range(int(np.log10(lo)), int(np.log10(hi)) + 1)
                 for m in (1, 3) if lo <= m * 10.0 ** e <= hi]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlim(-0.6, len(cams) - 0.4)
    ax.set_ylabel("shift (px)")
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    es._badge_ticks(ax, list(range(len(cams))), cams,
                    highlight=[highlight] if highlight is not None else [], xlabel=None)


def draw_eps(ax: Any, phase3: Mapping[int, float], phase4: Mapping[int, float] | None,
             highlight: int | None) -> None:
    """The telecentric depth term ε in 1/m, signed, as Phase 4 shows it (ED2-4: one unit and
    one sign everywhere): Phase 3 hollow, Phase 4 (when given) filled, a line at zero.
    1/m reads as % size change per cm of depth; negative: further away draws larger."""
    t = _tokens(ax)
    cams = sorted(set(phase3) | set(phase4 or {}))
    for k, cam in enumerate(cams):
        colour = es.camera_colour(cam, ax)
        a = phase3.get(cam, np.nan)
        b = (phase4 or {}).get(cam, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            ax.plot([k, k], [a, b], color=t["grid"], linewidth=2.0, zorder=1)
        if np.isfinite(a):
            ax.scatter([k], [a], s=60, facecolors="none", edgecolors=colour, linewidths=1.8,
                       zorder=3)
        if np.isfinite(b):
            ax.scatter([k], [b], s=60, zorder=4, **es.camera_mark_kwargs(cam, ax, "dot"))
    values = [v for v in list(phase3.values()) + list((phase4 or {}).values()) if np.isfinite(v)]
    ax.axhline(0, color=t["text"], linewidth=0.8, zorder=2)
    if values:
        low, high = min(values + [0.0]), max(values + [0.0])
        pad = 0.18 * max(high - low, 1e-6)
        ax.set_ylim(low - pad, high + pad)
    ax.set_xlim(-0.6, len(cams) - 0.4)
    ax.set_ylabel("ε (1/m)")
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    _plain_value_axis(ax)
    es._badge_ticks(ax, list(range(len(cams))), cams,
                    highlight=[highlight] if highlight is not None else [], xlabel=None)


#: Short two-line names for the error-ladder rungs (UX4-3), used as y-tick labels.
_RUNG_TICKS = {
    "Phase 2 · each camera alone": "Phase 2\neach alone",
    "Phase 3 · one rig": "Phase 3\none rig",
    "Phase 4 · target refined too": "Phase 4\ncube refined",
}


def _rung_tick(label: str) -> str:
    return _RUNG_TICKS.get(label, label)


def draw_error_ladder(ax: Any, ladder: Any) -> None:
    """Each camera's RMS error alone (Phase 2) and in the one rig (Phase 3), on the shared log
    px axis, with the Phase 4 mean when a Phase 4 run continued from this one (ED-4)."""
    t = _tokens(ax)
    scale = es.error_scale(ax)
    # SCI2-4 / ED2-13: one statistic (RMS per camera) on every rung; the Phase 4 rung is that
    # run's per-camera RMS replayed from its saved result, never its mean.
    rungs = ([("Phase 2 · each camera alone", ladder.phase2_rms)] if ladder.phase2_rms else []) \
        + [("Phase 3 · one rig", ladder.phase3_rms)]
    if getattr(ladder, "phase4_rms", None):
        rungs.append(("Phase 4 · target refined too", ladder.phase4_rms))
    ys = list(range(len(rungs)))
    # UX3-15: a slope chart. Within each rung the cameras sit on their own row (by camera
    # number), so 8 dots never merge into one smear, and a thin line in the camera's colour
    # links each camera from rung to rung.
    cams = sorted({cam for _l, values in rungs for cam in values})
    spread = 0.48
    offset = {cam: (k / max(len(cams) - 1, 1) - 0.5) * spread for k, cam in enumerate(cams)}
    for cam in cams:
        pts = [(values[cam], y + offset[cam]) for y, (_l, values) in zip(ys, rungs)
               if cam in values and np.isfinite(values[cam])]
        if len(pts) > 1:
            style = {k: v for k, v in es.camera_mark_kwargs(cam, ax, "line").items()
                     if k != "linewidth"}
            ax.plot(*np.array(pts).T, linewidth=1.0, alpha=0.55, zorder=2, **style)
    for y, (_label, values) in zip(ys, rungs):
        if values:
            vals = np.array(list(values.values()), dtype=float)
            ax.plot([vals.min(), vals.max()], [y, y], color=t["grid"], linewidth=4,
                    solid_capstyle="round", zorder=1, alpha=0.7)
            for cam, value in values.items():
                ax.scatter([value], [y + offset.get(cam, 0.0)], s=46, zorder=3,
                           **es.camera_mark_kwargs(cam, ax, "dot"))
    # Round 10 (stage 6 panel B): each trajectory ends at its last rung with its camera badge,
    # so the lines can be told apart where they run together (repelled, with leaders).
    if len(rungs) > 1:
        last_values = rungs[-1][1]
        end_cams = [cam for cam in cams if cam in last_values and np.isfinite(last_values[cam])]
        if end_cams:
            end_xy = np.array([(last_values[cam], ys[-1] + offset[cam]) for cam in end_cams])
            es.draw_camera_badges(ax, end_xy, end_cams, size=8, repel=True, zorder=5)
    # UX4-3: the rung names are the y-tick labels, outside the plot, so no point or line is
    # drawn under them (they used to sit on the data at the left of the axes).
    ax.set_yticks(ys)
    # UX8-3: the rung names are the only key to the dot rows; body size and colour at 1280 px
    ax.set_yticklabels([_rung_tick(label) for label, _v in rungs], fontsize=10,
                       color=t["text"], linespacing=1.1)
    # Round 10: the value ticks were the smallest text in panel B at 1280 px
    ax.tick_params(axis="x", labelsize=es.LABEL_SIZE)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(ys[-1] + 0.36, -0.75)
    scale.log_axis(ax, "x")
    values = [v for _l, d in rungs for v in d.values()]
    lo = min(values + [1.0]) / 2.5
    hi = max(values + [1.0]) * 2.5
    ax.set_xlim(max(lo, scale.lo), min(hi, scale.hi))
    ax.set_xlabel(es.error_label("rms", per="camera", short=True))
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def agreement_factor(agreement: Any, size: float, share: float = 0.12) -> int:
    """Exaggeration that makes the typical camera-alone difference (offset plus turn at the
    target's edge) about ``share`` of the target size; one of 1, 2, 3, 5, 10."""
    devs = [agreement.offset[c] + np.radians(agreement.angle_deg[c]) * 0.5 * size
            for c in agreement.alone]
    typical = float(np.median(devs)) if devs else 0.0
    best = 1
    for f in (1, 2, 3, 5, 10):
        if typical > 0 and f * typical <= share * size:
            best = f
    return best


def draw_agreement(ax: Any, agreement: Any, template: np.ndarray, up: np.ndarray,
                   mm_per_unit: float | None, factor: int = 1,
                   cameras: list[int] | None = None) -> None:
    """One photo, seen from the side farthest from every camera's axis: the target where each
    camera alone put it (thin outlines, camera colours; differences from the shared pose
    exaggerated ``factor``×) and the one pose they share (thick)."""
    from pyCamSet.workflow.education_visual_p3 import _exaggerate_rotation, target_edges
    t = _tokens(ax)
    origins, ends = target_edges(template)
    local = np.nanmean(np.asarray(template, dtype=float), axis=0)
    shared = agreement.shared
    axes = np.array(list(agreement.axes.values()))
    up = np.asarray(up, dtype=float)
    up = up / (np.linalg.norm(up) or 1.0)
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = ref - (ref @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    best, best_score = up, -np.inf
    for az in np.radians(np.arange(0, 360, 10)):
        for el in np.radians((20.0, 30.0)):
            d = np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up
            score = -float(np.max(np.abs(axes @ d))) if len(axes) else 0.0
            if score > best_score:
                best, best_score = d, score
    view = best
    right = np.cross(up, view)
    right /= np.linalg.norm(right) or 1.0
    vertical = np.cross(view, right)
    shared_centre = shared[:3, :3] @ local + shared[:3, 3]

    def shown(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R = _exaggerate_rotation(T[:3, :3], shared[:3, :3], factor, limit_deg=40.0)
        centre = shared_centre + factor * ((T[:3, :3] @ local + T[:3, 3]) - shared_centre)
        return R, centre

    def project(R: np.ndarray, centre: np.ndarray, pts: np.ndarray) -> np.ndarray:
        world = (pts - local) @ R.T + centre - shared_centre
        return np.column_stack([world @ right, world @ vertical])

    unit = mm_per_unit if mm_per_unit else 1.0
    cams = sorted(agreement.alone) if cameras is None else \
        [c for c in sorted(agreement.alone) if c in set(cameras)]
    # UX2-25 / ED2-12: every camera-alone outline the same weight, the shared pose in the
    # theme accent with its own dash (never a camera's colour).
    for cam in cams:
        R, centre = shown(agreement.alone[cam])
        a, b = project(R, centre, origins), project(R, centre, ends)
        style = {k: v for k, v in es.camera_mark_kwargs(cam, ax, "line").items()
                 if k != "linewidth"}
        for p, q in zip(a, b):
            ax.plot(*(np.stack([p, q]) * unit).T, linewidth=1.6, alpha=0.9, zorder=3, **style)
    # Round 10 (stage 6 C, Dark): on dark axes the accent (cyan) is too close to camera 6's
    # slate, so the shared pose is drawn in the neutral text colour there.
    shared_colour = t["text"] if es.is_dark_axes(ax) else t["accent"]
    a, b = project(shared[:3, :3], shared_centre, origins), \
        project(shared[:3, :3], shared_centre, ends)
    for p, q in zip(a, b):
        ax.plot(*(np.stack([p, q]) * unit).T, color=shared_colour, linewidth=2.4,
                linestyle=(0, (4, 2)), zorder=4, solid_capstyle="round")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], linewidth=2.0, **{k: v for k, v in es.camera_mark_kwargs(
        c, ax, "line").items() if k != "linewidth"}) for c in cams]
    handles.append(Line2D([], [], color=shared_colour, linewidth=2.4, linestyle=(0, (4, 2))))
    ax.legend(handles, [es.camera_badge_label(c) for c in cams] + ["shared"],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=len(handles), fontsize=9,
              frameon=False, handlelength=1.1, columnspacing=0.6, handletextpad=0.3,
              borderaxespad=0.1)


def draw_spreads(ax: Any, spreads: Any, mm_known: bool, *,
                 position_level_mm: float | None = POSITION_LEVEL_MM,
                 judge_position: bool = True, unchecked: Sequence[int] = ()) -> None:
    """Did the cameras agree? Each camera's first-guess spread across photos relative to
    camera 1. With ``judge_position``: position and turn on one axis as a share of their
    warning levels (UX-1; the position level scaled to the target, SCI2-5). Without (a
    telecentric rig, whose photos' depth is not seen): the turn alone, in degrees, against
    its 5° level; the position is not judged and not drawn. Cameras in ``unchecked`` (too few
    photos for the photo-outlier rule, SCI9-2 / ED9) have their turn bar hatched and muted,
    so they are not drawn as judged."""
    from matplotlib.patches import Patch
    t = _tokens(ax)
    cams = [int(item.camera) for item in spreads]
    x = np.arange(len(cams))
    unchecked = {int(c) for c in unchecked}

    def mute(bar: Any) -> None:
        bar.set_facecolor(t["muted"])
        bar.set_edgecolor(t["muted"])
        bar.set_alpha(0.35)
        bar.set_hatch("///")
        bar.set_linewidth(0.8)

    unchecked_handle = [Patch(facecolor=t["muted"], alpha=0.35, hatch="///",
                              edgecolor=t["muted"], label="too few photos to check")]
    if not judge_position:
        rot = np.array([float(item.rotation_scatter_deg) for item in spreads])
        bars = ax.bar(x, rot, 0.62, zorder=3, color=[es.camera_colour(c, ax) for c in cams],
                      linewidth=0)
        for k, cam in enumerate(cams):
            if cam in unchecked:
                mute(bars[k])
        style = es.limit_style(ax, es.limit_crossed(rot, ROTATION_LEVEL_DEG))
        ax.axhline(ROTATION_LEVEL_DEG, **style)
        ax.set_xlim(-0.6, len(cams) - 0.4)
        ax.set_ylim(0, max(float(np.nanmax(rot)) if rot.size else 0.0,
                           ROTATION_LEVEL_DEG) * 1.3)
        ax.set_ylabel("turn spread (°)")
        ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
        # round 9: the warning level is named in the legend, not as a label over the last bar
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=style["color"], linestyle=(0, (5, 3)), linewidth=1.4,
                          label=f"warning level {ROTATION_LEVEL_DEG:g}°")]
        if unchecked:
            handles.append(unchecked_handle[0])
        ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False,
                  borderaxespad=0.3)
        es._badge_ticks(ax, list(x), cams, xlabel=None)
        return
    level = float(position_level_mm or POSITION_LEVEL_MM)
    pos_mm = [item.position_scatter_mm if mm_known else None for item in spreads]
    pos = np.array([v / level if v is not None else np.nan for v in pos_mm], dtype=float)
    rot = np.array([float(item.rotation_scatter_deg) / ROTATION_LEVEL_DEG for item in spreads])
    w = 0.36
    for k, cam in enumerate(cams):
        if np.isfinite(pos[k]):
            ax.bar(x[k] - w / 2, 100 * pos[k], w, color=es.camera_colour(cam, ax), alpha=0.42,
                   hatch="///", edgecolor=es.camera_colour(cam, ax), linewidth=0.8,
                   zorder=3)
        turn = ax.bar(x[k] + w / 2, 100 * rot[k], w, zorder=3,
                      **es.camera_mark_kwargs(cam, ax))
        if cam in unchecked:
            mute(turn[0])
    style = es.limit_style(ax, es.limit_crossed(np.concatenate([100 * pos, 100 * rot]), 100.0))
    ax.axhline(100, **style)
    ax.annotate("warning level", (1.0, 100), xycoords=("axes fraction", "data"),
                xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom",
                fontsize=9, color=style["color"])
    top = np.nanmax(np.concatenate([100 * pos, 100 * rot, [100.0]]))
    ax.set_xlim(-0.6, len(cams) - 0.4)
    ax.set_ylabel("% of warning level")
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    handles = ([Patch(facecolor=t["muted"], alpha=0.42, hatch="///", edgecolor=t["muted"],
                      label="position")] if np.isfinite(pos).any() else []) + [
        Patch(facecolor=t["muted"], label="turn")] + (unchecked_handle if unchecked else [])
    ax.set_ylim(0, max(top, 100.0) * 1.45)
    ax.legend(handles=handles, loc="upper left", ncols=len(handles), fontsize=9,
              frameon=False, borderaxespad=0.3)
    es._badge_ticks(ax, list(x), cams, xlabel=None)


def draw_residual_density(ax: Any, visual: Any, limit: float) -> Any:
    """Every final residual (dx, dy) as a density, plus one 1-σ ellipse per camera (UX-30)."""
    t = _tokens(ax)
    xy = np.asarray(visual.final_residual_xy, dtype=float)
    keep = np.all(np.isfinite(xy), axis=1)
    # UX3-3: a density is a count (corners per cell): the amount map on a log count scale.
    from matplotlib.colors import LogNorm
    cells = ax.hexbin(xy[keep, 0], xy[keep, 1], gridsize=28,
                      extent=(-limit, limit, -limit, limit), cmap=es.amount_cmap(ax), mincnt=1,
                      norm=LogNorm(), linewidths=0, zorder=2)
    cells._edu_quantity = "amount"
    from matplotlib.patches import Ellipse
    for cam in range(visual.n_cameras):
        pts = xy[keep & (visual.rows_camera == cam)]
        if len(pts) < 5:
            continue
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        angle = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
        ax.add_patch(Ellipse(tuple(pts.mean(axis=0)), 2 * np.sqrt(max(vals[1], 0)),
                             2 * np.sqrt(max(vals[0], 0)), angle=angle, fill=False, zorder=4,
                             **{k: v for k, v in es.camera_mark_kwargs(cam, ax, "line").items()
                                if k != "linewidth"}, linewidth=1.8))
    ax.axhline(0, color=t["grid"], linewidth=0.8)
    ax.axvline(0, color=t["grid"], linewidth=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("x error (px)")
    ax.set_ylabel("y error (px)")
    return cells


# ======================================================================================
# 3D: one target in the rig (any education_scene3d scene: Qt widget or off-screen PyVista)


def build_rig_scene(scene: Any, visual: Any, motion: Mapping[str, np.ndarray], *, state: int,
                    level: int, camera: int | None, photo: int | None, theme: str,
                    labels: list[str] | None = None, trails: bool = False) -> dict[str, Any]:
    """(Re)build ``scene``: the target drawn once (its outline and its corners coloured by each
    corner's mean error over every photo, on the shared log scale), the cameras around it
    (``cam<i>``), the chosen photo's target as a ghost outline (``ghost``) and optional camera
    trails (``trail<i>``). The view is framed on the saved result, so it stays the same while
    the solver plays; then ``state`` is shown. Returns what a player needs."""
    from pyCamSet.workflow.education_visual_p3 import (
        corner_errors, posed_outline, target_edges, target_world_points)
    scene.clear()
    scale = es.error_scale(theme)
    t = es.stage_tokens(theme)
    last = len(visual.states) - 1
    centre, size = _target_geometry(visual)
    if visual.telecentric:
        glyph = 0.30 * size
    else:  # UX2-8: frusta about 12 % of the camera-to-target distance
        reach = np.linalg.norm(np.asarray(motion["C"][last], dtype=float) - centre, axis=1)
        reach = reach[np.isfinite(reach)]
        glyph = 0.12 * float(np.median(reach)) if reach.size else 0.30 * size
    for index in range(visual.n_cameras):
        R = motion["R"][last, index]
        C = motion["C"][last, index]
        scene.add_camera(f"cam{index}", index, R, -R @ C, scale=glyph,
                         telecentric=visual.telecentric,
                         label=labels[index] if labels else None)
    world = target_world_points(visual)
    origins, ends = target_edges(world)
    scene.add_rays(origins, ends, name="outline", colour=t["muted"], width=1.4, opacity=0.8)
    errors = corner_errors(visual, last)
    seen = np.isfinite(errors)
    scene.add_points(world[seen], errors[seen], name="corners", scalar_bar="px", size=9,
                     **scale.scene_kwargs())
    if (~seen).any():
        scene.add_points(world[~seen], name="unseen", colour=t["muted"], size=4)
    ghost = None
    # ED3-5 / UX3-7: the chosen photo's target outline is a second cube in another place; it
    # is drawn from L2 only, in the accent (the dots are the reference photo's target), and
    # the heading names both.
    if (level >= 2 and photo is not None and photo != visual.fixed_pose
            and motion["posed"][int(photo)]):
        a, b = posed_outline(visual, motion["pose_R"][last, photo], motion["pose_C"][last, photo])
        scene.add_rays(a, b, name="ghost", colour=t["accent"], width=1.8, opacity=0.7)
        ghost = int(photo)
    if trails:
        # UX2-6 / ED2-14: each camera's path as the shared fading trail with an arrowhead,
        # a ghost of the camera where it started, grown with the playback position.
        for index in range(visual.n_cameras):
            path = motion["C"][:, index]
            R0 = motion["R"][0, index]
            scene.add_trail(trail_points(path, state), name=f"trail{index}", camera=index,
                            start_pose=(R0, -R0 @ path[0]), telecentric=visual.telecentric,
                            scale=glyph)
    _fit_rig_view(scene, visual, camera)
    if state != last:
        scene.apply_frame(rig_frame(visual, motion, state, ghost=ghost))
    return {"ghost": ghost}


def trail_points(path: np.ndarray, position: float) -> np.ndarray:
    """A camera's drawn path up to a (possibly fractional) playback position: the recorded
    states passed, plus the interpolated point between the last two (display only)."""
    path = np.asarray(path, dtype=float)
    position = float(np.clip(position, 0, len(path) - 1))
    k = int(np.floor(position + 1e-9))
    points = path[:k + 1]
    s = position - k
    if s > 1e-6 and k + 1 < len(path):
        points = np.vstack([points, (1 - s) * path[k] + s * path[k + 1]])
    return points


def _fit_rig_view(scene: Any, visual: Any, camera: int | None) -> None:
    """Frame the rig. A flat board (pinhole example, UX2-8) is never shown edge-on: the view
    keeps the board's normal 30-50° from the viewing direction, choosing among those views
    the one that best separates the cameras."""
    from pyCamSet.workflow.education_visual_p3 import target_world_points
    pts = target_world_points(visual)
    pts = pts[np.all(np.isfinite(pts), axis=1)] if len(pts) else pts
    flat = False
    normal = None
    if len(pts) >= 3:
        centred = pts - pts.mean(axis=0)
        _u, s, vt = np.linalg.svd(centred, full_matrices=False)
        flat = s[-1] <= 1e-6 * max(s[0], 1e-12)
        normal = vt[-1]
    if not flat or normal is None:
        scene.fit_view(selected=camera)
        return
    up = tuple(float(v) for v in visual.up)
    from pyCamSet.gui.education_scene3d import _view_direction, choose_view
    centres = np.array(list(scene.camera_centres().values())) if scene.camera_centres() else \
        np.zeros((0, 3))
    focus = pts.mean(axis=0)
    names = list(scene.camera_centres())
    sel = names.index(f"cam{camera}") if camera is not None and f"cam{camera}" in names else None
    best, best_score = None, -np.inf
    for el in (14, 22, 30, 38, 46, 54, 62):
        for az in np.arange(0.0, 360.0, 10.0):
            d = _view_direction(az, el, up)
            angle = np.degrees(np.arccos(min(1.0, abs(float(d @ normal)))))
            if not 30.0 <= angle <= 50.0:
                continue
            # separation score from choose_view on this one direction (+ facing the board)
            az_el = choose_view(centres if len(centres) else focus[None], focus, up=up,
                                selected=sel, elevations=(el,), azimuths=(az,))
            rel = (centres - focus) if len(centres) else np.zeros((1, 3))
            right = np.cross(up, d)
            right /= np.linalg.norm(right) or 1.0
            upv = np.cross(d, right)
            xy = np.stack([rel @ right, rel @ upv], axis=1)
            extent = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-9)
            diff = xy[:, None] - xy[None]
            dist = np.hypot(diff[..., 0], diff[..., 1])
            dist[np.diag_indices(len(xy))] = np.inf
            score = float(dist.min()) / extent if len(xy) > 1 else 0.0
            # Look at the face the cameras see (eye on the cameras' side of the board).
            score += 0.3 * float((d @ normal) * (float(np.mean(rel, axis=0) @ normal)) > 0)
            if score > best_score:
                best, best_score = az_el, score
    if best is None:
        scene.fit_view(selected=camera, up=up)
    else:
        scene.fit_view(selected=camera, up=up, azimuth=best[0], elevation=best[1])


def _target_geometry(visual: Any) -> tuple[np.ndarray, float]:
    from pyCamSet.workflow.education_visual_p3 import target_centre_size
    return target_centre_size(visual)


def rig_frame(visual: Any, motion: Mapping[str, np.ndarray], k: int, *,
              ghost: int | None) -> dict:
    """The scene frame of recorded state ``k`` (cameras, corner colours, the ghost outline)."""
    from pyCamSet.workflow.education_visual_p3 import (
        corner_errors, posed_outline, target_world_points)
    world = target_world_points(visual)
    seen = np.isfinite(corner_errors(visual, -1))
    frame: dict[str, dict[str, Any]] = {
        "cameras": {f"cam{i}": {"R": motion["R"][k, i],
                                "t": -motion["R"][k, i] @ motion["C"][k, i]}
                    for i in range(visual.n_cameras)},
        "points": {"corners": {"xyz": world[seen], "scalars": corner_errors(visual, k)[seen]}},
    }
    if ghost is not None:
        a, b = posed_outline(visual, motion["pose_R"][k, ghost], motion["pose_C"][k, ghost])
        frame["rays"] = {"ghost": {"origins": a, "ends": b}}
    return frame


def rig_frames(visual: Any, motion: Mapping[str, np.ndarray], *, ghost: int | None) -> list[dict]:
    """One scene frame per recorded state for :class:`ScenePlayer` (cameras, corner colours
    and the ghost outline move; trails and the target outline stay)."""
    return [rig_frame(visual, motion, k, ghost=ghost) for k in range(len(visual.states))]


# ======================================================================================
# Qt helpers


class _FigurePane(QWidget):
    """A themed matplotlib canvas that redraws from its drawer on a theme switch."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        perf.cap_canvas_pixel_ratio(self.canvas)
        self.canvas.setParent(self)
        self.canvas.setMinimumSize(160, 120)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setAccessibleName("Stage side figure")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self._drawer = None

    def draw(self, drawer) -> Figure:
        import matplotlib
        theme = es.active_theme_name()
        self._drawer = drawer
        self.figure.clear()
        self.figure._edu_theme = theme
        with matplotlib.rc_context(es.figure_style(theme)):
            self.figure.set_facecolor(es.stage_tokens(theme)["stage"])
            es.align_panel_titles(self.figure)
            drawer(self.figure, theme)
        self.canvas.draw_idle()
        return self.figure

    def set_theme(self, _name: str) -> None:
        if self._drawer is not None:
            self.draw(self._drawer)


# The 3D panel heading is the shared one (same letter spacing and baseline as the
# matplotlib panel titles, UX2-19); kept under its old name for this module.
from pyCamSet.gui.education_stage import PanelHeading as _PanelHeading  # noqa: E402


class _SceneAndFigure(QWidget):
    """A 3D scene (left, with a heading) beside a figure (right); stretch set per stage."""

    def __init__(self, scene: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.heading = _PanelHeading(self)
        self.pane = _FigurePane(self)
        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        left_layout.addWidget(self.heading)
        left_layout.addWidget(scene, 1)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._layout.addWidget(left, 5)
        self._layout.addWidget(self.pane, 5)
        self._left = left

    def set_ratio(self, left: int, right: int) -> None:
        self._layout.setStretch(0, left)
        self._layout.setStretch(1, right)

    def set_theme(self, name: str) -> None:
        setter = getattr(self.scene, "set_theme", None)
        if callable(setter):
            setter(name)
        self.heading.restyle()
        self.pane.set_theme(name)


# ======================================================================================
# The view


class Phase3View(QWidget):
    """Phase 3 storyboard on the shared stage layout (see the module docstring)."""

    def __init__(self, context: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self.run_context = self._run_context(context)
        self.metadata = _plain(context.metadata)
        self._learning_level = 1
        # The data below is written by _prepare_data on the worker thread, and read by the GUI
        # thread only after _prepared (the worker's ``ready`` signal). Until then the view shows
        # the 'Preparing the walkthrough' state.
        self._ready = False
        self._prepare_error: str | None = None
        self._prepare_status = "Preparing the walkthrough ..."
        self._derivation: Any = None
        self._trace: Any = None
        self._replay_handler = None
        self._final_params: np.ndarray | None = None
        self._user_camera = False  # the learner picked a camera (stages stop choosing one)
        self._playback: Any = None
        self._single_face: dict[str, Any] | None = None
        self._visual: Any = None
        self._visual_reason: str | None = None
        self._evidence: Any = None
        self._join_events: Any = ()
        # Caches for the pictures (all recomputed from the saved run, never written back).
        self._camsets: Any = None
        self._lens: list | None = None
        self._agreement_cache: Any = None
        self._ladder_cache: Any = None
        self._motion_cache: dict[float, dict[str, np.ndarray]] = {}
        self._factor: int | None = None
        self._photo_source = None
        # Byte-capped: a full-resolution photo per camera-photo must not accumulate.
        self._photos: ByteLRU = ByteLRU()
        self._filling_photos = False
        # The stage list until the data says whether the single-face and lens stages exist.
        self._stage_keys = [STAGE_GRAPH, STAGE_FIRST, STAGE_SOLVER, STAGE_RESULT]
        self._scene = None
        self._split: _SceneAndFigure | None = None
        self._player = None
        self._solver_artists: dict[str, Any] = {}
        self._schur_cache: dict[int, Any] = {}
        self._structure = None
        self._cursor_clock = QElapsedTimer()
        self._cursor_clock.start()
        # UX3-6 / P3-1: the 2D panels redraw at most every CURSOR_INTERVAL_MS; a frame that
        # comes sooner is deferred (drawn once the interval has passed), never dropped.
        self._pending_position: float | None = None
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.timeout.connect(self._draw_pending_position)
        self._current_key = None
        self._buildup: Any = None
        self.transport: Any = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.stage_view = EducationStageView([STAGE_TITLES[key] for key in self._stage_keys],
                                             self, phase_label="Phase 3")
        layout.addWidget(self.stage_view)
        self.stage_view.stageChanged.connect(self._show_stage)
        self.stage_view.levelChanged.connect(self._level_changed)
        self._worker = EducationWorker(self)
        self._worker.ready.connect(self._prepared)
        self._worker.failed.connect(self._prepare_failed)
        self._worker.status.connect(self._prepare_progress)
        self._worker.submit("phase3", self._prepare_data)
        self._show_stage(0)

    # -- preparation: _prepare_data runs on the worker thread, the rest on the GUI thread ----

    def is_ready(self) -> bool:
        """True once the derivation, the replay and the pictures' data are prepared."""
        return self._ready

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Block until the data is prepared (tests and screenshot tools only)."""
        self._worker.wait_until_idle(timeout)
        return self.is_ready()

    def close_background(self) -> None:
        """Cancel any preparation still running (called when the dialog closes)."""
        self._worker.close()

    def _prepare_data(self, report: Callable[[str], None]) -> tuple[bool, Any]:
        """The saved run's derivation, replay and picture data (no Qt; worker thread)."""
        from pyCamSet.workflow.education_visual_p3 import join_events, most_improved_camera
        report("Reading the saved run ...")
        self._derivation = derive_phase3(self.run_context)
        self._trace = read_trace(self.context.run_dir, self.context.run_id, "phase3")
        report("Recalculating the first guess and the solver steps. This can take a few seconds ...")
        self._playback = self._make_playback()
        self._single_face = self._single_face_record()
        # The pictures' data first: the agreement rule in the evidence reads the camera model.
        self._visual, self._visual_reason = self._make_visual()
        self._evidence = self._make_evidence()
        self._join_events = join_events(self._trace, self.metadata.get("diagnostics"))
        report("Preparing the pictures ...")
        has_lens = bool(self._visual is not None and self._lens_changes())
        start_camera = most_improved_camera(self._visual) if self._visual is not None else None
        # Warm the caches the stages read (the first guess, the joins, the solver's rig motion,
        # the agreement and the error ladder), so no stage recomputes them on the GUI thread.
        # The view is not ready yet, so nothing else reads these attributes meanwhile.
        if self._visual is not None:
            report("Preparing the solver's motion and the error comparisons ...")
            self._warm(self._seed, self._join_candidates, self._motion_factor)
            self._warm(lambda: self._motion(float(self._motion_factor())), lambda: self._motion(1.0))
            self._warm(self._agreement, self._ladder, self._lens_effect)
        if self._recorded() and self._visual is not None:
            # The first use of the numba solver kernel compiles it (once per machine when the
            # cache folder is kept); do it here, with a message, not on the GUI thread.
            report("Preparing the solver kernels (the first run on this machine can take a minute) ...")
            last = len((self._playback.get("arrays") or [None])) - 1
            self._warm(lambda: self._schur(last))
        return has_lens, start_camera

    @staticmethod
    def _warm(*steps: Callable[[], Any]) -> None:
        """Run each cache-filling step; a step that fails leaves its cache empty, as before."""
        for step in steps:
            try:
                step()
            except Exception:  # noqa: BLE001 - the stage then reports the gap itself
                pass

    def _prepared(self, _key: Any, data: tuple[bool, Any]) -> None:
        """Build the pickers, the transport and the stage list now that the data exists."""
        has_lens, start_camera = data
        from pyCamSet.gui.education_stage import camera_picker, photo_picker
        from PySide6.QtWidgets import QComboBox

        keys = [STAGE_GRAPH, STAGE_FIRST]
        if self._single_face is not None:
            keys.append(STAGE_SINGLE)
        keys.append(STAGE_SOLVER)
        if has_lens:
            keys.append(STAGE_LENS)
        keys.append(STAGE_RESULT)
        self._stage_keys = keys
        if keys != [STAGE_GRAPH, STAGE_FIRST, STAGE_SOLVER, STAGE_RESULT]:
            # Selects stage 0 and emits stageChanged; the view is not ready yet, so it only
            # shows the pending state for that stage.
            self.stage_view.set_stages([STAGE_TITLES[key] for key in self._stage_keys])
        # One camera picker (badge + "Camera n · name") and one photo picker (1-based).
        names = self._camera_names()
        self.camera_picker = camera_picker(dict(enumerate(names)), current=start_camera)
        self.camera_picker.setToolTip("The camera whose photo, badge and dots are followed.")
        # UX2-3: the camera picker shows its full name (short stage pills leave the room); the
        # photo picker stays narrow ("Photo 88").
        policy = QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        self.stage_view.add_picker("", self.camera_picker)
        self.photo_picker = photo_picker([])
        self.photo_picker.setAccessibleName("Photo of the chosen camera")
        self.photo_picker.setSizeAdjustPolicy(policy)
        self.photo_picker.setMinimumContentsLength(8)
        self.stage_view.add_picker("", self.photo_picker)
        self._fill_photo_picker()
        recovered, _ = self._single_face_entries()
        self.single_face_image = photo_picker(
            [int(entry.get("image")) for entry in recovered],
            details={int(entry.get("image")): f"{_px(entry.get('rms_px'))} px fit"
                     for entry in recovered})
        self.single_face_image.setAccessibleName("Recovered single-face image")
        self.single_face_image.setToolTip("Choose a recovered photo to see its candidate tilts' "
                                          "fits. Photos are numbered from 1.")
        for row, entry in enumerate(recovered):  # the saved teaching record is keyed by index
            self.single_face_image.setItemData(row, str(entry.get("image")))
        if not recovered:
            self.single_face_image.addItem("No recovered photo")
        self.single_face_image.setEnabled(bool(recovered))
        self.stage_view.add_picker("", self.single_face_image)
        # The camera joins: the shared transport bar (UX-3), ending on the whole graph.
        from pyCamSet.gui.education_dialog import TransportBar
        self.transport = (TransportBar(len(self._join_events) + 1, self,
                                       describe=self._describe_join, short=self._short_join,
                                       unit_label="Camera joins", interval_ms=1100,
                                       min_interval_ms=400, start=len(self._join_events))
                          if self._join_events else None)
        self._buildup = self.transport  # older name (review tools)
        if self.transport is not None:
            self.transport.hide()
            self.transport.positionChanged.connect(
                lambda _p: self._current_key == STAGE_GRAPH and self._redraw_graph())

        self.camera_picker.currentIndexChanged.connect(self._camera_changed)
        self.photo_picker.currentIndexChanged.connect(self._photo_changed)
        self.single_face_image.currentIndexChanged.connect(
            lambda _i: self._current_key == STAGE_SINGLE and self._show_stage(
                self.stage_view.current_stage()))
        self._ready = True
        self._show_stage(self.stage_view.current_stage())

    def _prepare_failed(self, _key: Any, message: str) -> None:
        self._prepare_error = message
        self._show_stage(self.stage_view.current_stage())

    def _prepare_progress(self, _key: Any, text: str) -> None:
        self._prepare_status = text
        if not self._ready and self._prepare_error is None:
            self._show_pending(self.stage_view.current_stage())

    def _show_pending(self, index: int) -> None:
        """The stage frame while the data is prepared: a title, the stepper and this text."""
        self._current_key = None
        self.stop_for_context_change()
        self.stage_view.set_glossary(None)
        if self._prepare_error is not None:
            title = "The walkthrough could not be prepared"
            message = (f"The walkthrough could not be prepared. Not shown: "
                       f"{self._prepare_error}.")
        else:
            title = "Preparing the walkthrough"
            message = f"{self._prepare_status} The stages fill in when this is ready."
        self.stage_view.setAccessibleDescription(message)
        self.stage_view.set_chips([])
        self.stage_view.set_caption(message)
        self.stage_view.set_controls(None)
        self.stage_view.set_numbers((), ())
        self.stage_view.set_busy(None if self._prepare_error is not None
                                 else self._prepare_status.rstrip(" ."))

        def draw(fig: Any, _theme: str) -> None:
            ax = fig.add_subplot()
            es.empty_panel(ax, message)
            es.panel_title(ax, None, title, "the stages fill in when the data is ready")
        self.stage_view.draw_figure(draw)

    # -- navigation (the dialog calls these) ---------------------------------------------

    def step_back(self) -> None:
        self.stage_view.stepper.previous()

    def step_forward(self) -> None:
        self.stage_view.stepper.next()

    def stop_for_context_change(self) -> None:
        """Stop the join build-up and the solver playback (dialog: run/close changes)."""
        if self._buildup is not None:
            self._buildup.stop_for_context_change()
        if self._player is not None:
            self._player.stop_for_context_change()

    @property
    def _frames(self) -> list[dict[str, Any]]:
        """The saved-evidence blocks (older callers read them under this name)."""
        return self._evidence

    def stage_keys(self) -> list[str]:
        """The stage keys in order (``graph``, ``first``, [``single``,] ``solver``, ``result``)."""
        return list(self._stage_keys)

    def show_stage(self, key: str) -> None:
        """Select a stage by key."""
        self.stage_view.set_current_stage(self._stage_keys.index(key))

    # -- data ----------------------------------------------------------------------------

    def _camera_names(self) -> list[str]:
        if self._visual is not None:
            return list(self._visual.camera_names)
        result = self._derivation
        names = [name for name, _ in getattr(result, "per_camera_residual_mean_px", ()) or ()]
        return [str(name) for name in names]

    def _camera_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self._camera_names())}

    def _selected_camera(self) -> int | None:
        data = self.camera_picker.currentData()
        return int(data) if data is not None else None

    def _make_visual(self) -> tuple[Any, str | None]:
        """Recompute every picture's data from the saved vectors (recorded steps when valid)."""
        from pyCamSet.workflow.education_visual_p3 import build_phase3_visual
        payload = self._playback
        try:
            if payload.get("status") == "available" and self._replay_handler is not None:
                info = payload.get("payload") or {}
                self._final_params = np.asarray(payload["arrays"][-1], dtype=np.float64)
                return build_phase3_visual(
                    self._replay_handler, payload["arrays"], payload["nits"], payload["costs"],
                    source="recorded steps", loss=str(info.get("loss", "linear")),
                    f_scale=float(info.get("f_scale", 1.0)),
                    solver=str(info.get("solver", ""))), None
            artifact = self.run_context.artifact("phase3")
            initial_path = Path(self.context.run_dir) / "phase3_initial_state.npz"
            if artifact is None or not Path(artifact).is_file() or not initial_path.is_file():
                return None, "the saved result or its starting point is unavailable"
            from pyCamSet.utils.saving import load_CameraSet
            from pyCamSet.workflow.education_playback import prepare_saved_phase3_handler
            with np.load(initial_path, allow_pickle=False) as initial:
                start = np.asarray(initial["calibration_params"], dtype=np.float64)
            camera_set = load_CameraSet(artifact)
            saved = np.asarray(camera_set.calibration_params, dtype=np.float64)
            handler = camera_set.calibration_handler
            prepare_saved_phase3_handler(handler, start)
            self._replay_handler = handler
            self._final_params = saved
            options = (self.metadata.get("params") or {}).get("problem_options") or {}
            return build_phase3_visual(
                handler, [start, saved], [0, 1], [np.nan, np.nan],
                source="start and result only",
                solver=str(options.get("solver", "")) if isinstance(options, Mapping) else ""), None
        except Exception as exc:  # never break the stage: say why the pictures are missing
            return None, f"the saved geometry could not be rebuilt ({type(exc).__name__})"

    def _recorded(self) -> bool:
        return (self._visual is not None and self._visual.source == "recorded steps"
                and len(self._visual.states) >= 2)

    def _consistency(self) -> dict:
        if self._visual is None:
            return {}
        from pyCamSet.workflow.education_visual_p3 import consistency
        derivation = (self._derivation if not isinstance(self._derivation, DerivationUnavailable)
                      else None)
        return consistency(self._visual, derivation, self.metadata.get("diagnostics"))

    # -- saved evidence (the reviewed logic, unchanged apart from camera numbering) ---

    @staticmethod
    def _run_context(context: Any) -> Any:
        """Rebuild the Qt-free provenance context from the dialog's immutable view."""
        if callable(getattr(context, "verify_input", None)):
            return context
        from pyCamSet.workflow.education_trace import RunContext

        metadata = _plain(context.metadata)
        return RunContext.from_run(
            context.manager, {**metadata, "_run_dir": context.run_dir}, context.phase)

    def _lockbox_source_for_replay(self):
        settings = (self.metadata.get("params") or {}).get("lockbox") or {}
        if not isinstance(settings, Mapping):
            return None, None, "lockbox settings are malformed"
        if not settings.get("enabled", False):
            return None, None, None
        raw = settings.get("source_camset")
        digest = (self.metadata.get("artifacts") or {}).get("lockbox_source_sha256")
        if not isinstance(raw, str) or not raw or not isinstance(digest, str):
            return None, None, "lockbox source or saved digest is missing"
        source = Path(raw)
        if not source.is_absolute() or not source.is_file():
            return None, None, "lockbox source is not an available absolute file"
        try:
            if _file_sha256(source) != digest:
                return None, None, "lockbox source no longer matches the saved run"
        except OSError:
            return None, None, "lockbox source could not be checked"
        try:
            from pyCamSet.optimisation.camera_lockbox import CameraLockboxConfig
            config = CameraLockboxConfig(
                enabled=True,
                rotation_half_width=float(settings.get("rotation_half_width", 0.1)),
                translation_half_width=float(settings.get("translation_half_width", 0.1)),
                rotation_sigma=float(settings.get("rotation_sigma", 0.05)),
                translation_sigma=float(settings.get("translation_sigma", 0.01)),
                center_sigma=float(settings.get("center_sigma", 0.0)))
        except (TypeError, ValueError):
            return None, None, "saved lockbox settings cannot be reproduced"
        return config, source, None

    def _make_playback(self) -> dict[str, Any]:
        """Validate the real trace endpoints before exposing any accepted steps."""
        params = self.metadata.get("params") or {}
        recording = params.get("record_education")
        if recording is not True:
            return {"status": "unavailable", "reason": "steps not recorded for this run"}
        if self._trace.get("status") != "available":
            return {"status": "unavailable", "reason": self._trace.get("reason", "corrupt or invalid sidecar")}
        artifact = self.run_context.artifact("phase3")
        initial_path = self.context.run_dir / "phase3_initial_state.npz"
        if artifact is None or not artifact.is_file() or not initial_path.is_file():
            return {"status": "unavailable", "reason": "saved start or result is unavailable"}
        lockbox_config, lockbox_source, lockbox_reason = self._lockbox_source_for_replay()
        if lockbox_reason is not None:
            return {"status": "unavailable", "reason": lockbox_reason}
        try:
            from pyCamSet.utils.saving import load_CameraSet
            with np.load(initial_path, allow_pickle=False) as initial:
                expected = np.asarray(initial["calibration_params"], dtype=np.float64)
            camera_set = load_CameraSet(artifact)
            saved = np.asarray(camera_set.calibration_params, dtype=np.float64)
            # Rehydrate an active prior only from the saved, digest-checked
            # source. A replay without that exact source is unavailable.
            def fresh_handler():
                handler = load_CameraSet(artifact).calibration_handler
                # Keep the private replay handler: the pictures are evaluated on it.
                self._replay_handler = handler
                if lockbox_config is not None:
                    source_camset = load_CameraSet(lockbox_source)
                    if _file_sha256(lockbox_source) != (
                            (self.metadata.get("artifacts") or {}).get("lockbox_source_sha256")):
                        raise ValueError("lockbox source changed during replay")
                    handler.lockbox_config = lockbox_config
                    handler.lockbox_source_camset = source_camset
                    handler.lockbox_prior = None
                return handler

            frames = build_phase3_frames(
                fresh_handler, self._trace, expected_start=expected,
                saved_params=saved)
            return frames
        except Exception as exc:
            return {"status": "unavailable", "reason": f"recorded steps could not be read ({type(exc).__name__})"}

    def _make_evidence(self) -> list[dict[str, Any]]:
        """The saved evidence behind the stages (Numbers drawer, chips, captions)."""
        artifacts = self.metadata.get("artifacts") or {}
        params = self.metadata.get("params") or {}
        used = artifacts.get("phase2_initial_camset_used")
        phase2_edge = self.run_context.verify_input("phase2")
        promoted_origin = self._unmarked_promoted_phase2_origin(phase2_edge)
        origin = (promoted_origin if promoted_origin else
                  f"Phase 2 run {phase2_edge.parent_run_id}'s own saved result"
                  if phase2_edge.verification.value == "verified" and phase2_edge.parent_run_id
                  else f"file as it is now (not confirmed): {used or 'used input unavailable'}")
        decision = self._decision_frame_copy()
        frames = [
            {"title": "Input chain", "copy": (
                f"Run {self.context.run_id}. Lens estimates came from {origin}. "
                "The saved detections and optional lockbox prior are shown only from this run's recorded inputs. "
                + (("This unmarked promoted input is not confirmed as a trustworthy Phase 2 seed; "
                    "its current file is not proof of a clean Phase 2 result. "
                    "For a checkable comparison, select a verified Phase 2 run and rerun Phase 3.")
                   if promoted_origin else
                   "This input could not be verified; its current file is not proof of what the run used. "
                   "For a checkable comparison, select the recorded Phase 2 run and rerun Phase 3."
                   if phase2_edge.verification.value != "verified" else "")),
             "headers": ["Input", "Recorded value", "Evidence"],
             "rows": [("Phase 2 camset", str(used or "not recorded"), phase2_edge.verification.value),
                      ("Phase 1 detections", str(artifacts.get("phase1_detection_pickle_used") or "not recorded"),
                       "run record"),
                      ("Lockbox prior", str(artifacts.get("lockbox_prior", "not used / not recorded")), "run record")]},
            {"title": "First guess and solver start", "copy": self._seed_copy(),
             "detail": self._seed_detail(),
             "headers": ["State", "Value", "Evidence"], "rows": self._seed_rows()},
            {"title": "Same observations before and after", "copy": self._residual_copy(),
             "detail": self._residual_detail(),
             "headers": ["Camera / point", "Rows used", "Before mean (px)", "After mean (px)",
                         "Residual vector (before → after, px)"],
             "rows": self._residual_rows()},
            {"title": "Recorded-step playback", "copy": self._playback_copy(),
             "headers": ["Solver step", _objective_header(self._playback),
                         "Largest camera-centre displacement"],
             "rows": self._step_rows()},
            {"title": "Did the cameras agree?",
             **{key: decision[key] for key in ("copy", "detail", "expert") if key in decision},
             "headers": ["Camera index", "Position spread", "Rotation spread", "Usable photos", "Rule"],
             "rows": self._spread_rows()},
            {"title": "Quality and next action",
             "copy": " ".join((_status_copy(self.run_context), self._outlier_copy(),
                               _next_step_copy(self.run_context, decision.get("answer")))),
             "headers": ["Setting", "Recorded value", "Source"], "rows": [
                 ("Outlier exclusion", self._outlier_setting() or "unknown / not recorded", "Phase 3 solver options"),
                 ("Run status", self.context.status, "Phase 3 run record")]},
        ]
        if self._single_face is not None:
            # A seventh stage only for runs that opted in or left recoverable images out:
            # it is placed after the first guess.
            frames.insert(2, {"title": _SINGLE_FACE_TITLE, **self._single_face_copy(),
                              "headers": ["Image", "Outcome", "Best fit (px)", "Cameras", "Reason / evidence"],
                              "rows": self._single_face_rows()})
        return frames

    def _seed_copy(self) -> str:
        result = self._derivation
        if isinstance(result, DerivationUnavailable):
            return f"First guess unavailable: {result.reason}. The saved run is not changed."
        source = result.seed_camset_origin
        promoted_origin = self._unmarked_promoted_phase2_origin(
            self.run_context.verify_input("phase2"))
        if promoted_origin:
            source = promoted_origin
        copy = (f"The first guess of positions was worked out at the start of Phase 3 using lens estimates from {source}. "
                f"Its average error was {_number(result.seed_weighted_mean_px)} px. "
                f"Where the solver began is a separate state, with an average error of {_number(result.d35_initial_px)} px.")
        if result.weighted_seed_matches_d35 is False:
            copy += " These two averages could not be reconciled, so they are not compared."
        return copy

    def _seed_detail(self) -> str:
        """L2: name the saved diagnostics the plain first-guess numbers reproduce."""
        result = self._derivation
        if isinstance(result, DerivationUnavailable):
            return ""
        copy = ("The recalculated per-image values reproduce D3.3 (the seed error); their weighted mean is "
                f"{_number(result.seed_weighted_mean_px)} px. The solver starting point is D3.5, "
                f"{_number(result.d35_initial_px)} px.")
        if result.weighted_seed_matches_d35 is False:
            copy += (" The weighted mean does not reconcile with D3.5, so that comparison is "
                     "unavailable. A new Phase 3 run computes its own first guess and starts "
                     "from it.")
        elif result.weighted_seed_matches_d35 is True:
            copy += " The weighted mean reproduces D3.5 within the recorded tolerance."
        if len(result.seed_detection_count_by_image) == len(result.seed_mean_by_image_px):
            total = sum(result.seed_detection_count_by_image)
            copy += f" D3.3 includes {total} detection(s) across the posed images."
        return copy

    def _seed_rows(self) -> list[tuple[str, str, str]]:
        result = self._derivation
        if isinstance(result, DerivationUnavailable):
            return [("First guess / solver start", "unavailable", result.reason)]
        level = getattr(self, "_learning_level", 1)
        # A1 section 1: diagnostic names and conventions from L2 only.
        evidence = ("recalculated from saved results; checked against D3.3" if level >= 2
                    else "recalculated from saved results")
        # An image the first guess could not place has no seed error: say why instead of a bare
        # 'unavailable'. (On a real telecentric rig, 17 unposed images were listed that way.)
        excluded = dict(self._excluded_images())

        def _seed_row(index: int, value: Any) -> tuple[str, str, str]:
            missing = value is None or (isinstance(value, float) and not math.isfinite(value))
            if missing and str(excluded.get(index, "")).startswith("not posed"):
                return (f"{_photo_row(index)} mean error (px)", "not posed",
                        "the first guess could not place the target in this image; it was left out of the solve")
            return (f"{_photo_row(index)} mean error (px)", _number(value), evidence)
        rows = [_seed_row(index, value) for index, value in enumerate(result.seed_mean_by_image_px)]
        if level >= 2:
            first_guess = ("First guess, weighted mean (px)", result.d35_convention)
            start = ("Solver starting point (px)", "saved D3.5")
        else:
            first_guess = ("First guess, average error (px)",
                           "could not be reconciled with where the solver began"
                           if result.weighted_seed_matches_d35 is False else
                           "average over every detected point")
            start = ("Where the solver began, average error (px)", "saved with this run")
        if level >= 3:
            rows.extend(
                (f"{_photo_row(index)} detections (D3.3 N)", str(count),
                 "counted from the initialiser's persisted detection rows")
                for index, count in enumerate(result.seed_detection_count_by_image))
        if (level >= 3
                and len(result.seed_detection_count_by_image) == len(result.seed_mean_by_image_px)):
            total = sum(result.seed_detection_count_by_image)
            total_error = (result.seed_weighted_mean_px * total
                           if result.seed_weighted_mean_px is not None else None)
            rows.append(("D3.3 summed error, px (L3)", _number(total_error),
                         f"sum over N detections = weighted mean × N; N={total}"))
        return rows + [
                    (first_guess[0], _number(result.seed_weighted_mean_px), first_guess[1]),
                    (start[0], _number(result.d35_initial_px), start[1])]

    def _unmarked_promoted_phase2_origin(self, edge: Any) -> str | None:
        """Name a path-verified but scientifically untrusted promoted input."""
        if edge.verification.value != "verified" or not edge.parent_run_id:
            return None
        manager = getattr(self.run_context, "manager", None)
        finder = getattr(manager, "find_run", None)
        if not callable(finder):
            return None
        try:
            parent = finder("phase2", str(edge.parent_run_id))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(parent, Mapping):
            return None
        diagnostics = parent.get("diagnostics") or {}
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}
        promoted = diagnostics.get("promoted_from") or parent.get("promoted_from")
        education = parent.get("education") or {}
        if not isinstance(education, Mapping):
            education = {}
        marker = education.get("trial_camset_snapshot_version")
        if not promoted or type(marker) is int and marker == 1:
            return None
        return "an unmarked Optimisation-tab trial camset; it may contain later-stage cameras"

    def _paired_residuals(self):
        result = self._derivation
        if isinstance(result, DerivationUnavailable) or result.seed_observations is None:
            return None
        if result.seed_observation_status != "verified on the same ordered optimised rows":
            return None
        final = result.observations
        seed = result.seed_observations
        if len(seed) != len(final):
            return None
        identities = [(row.camera, row.image, row.key, row.observed_xy) for row in final]
        if len(set(identities)) != len(identities):
            return None
        if identities != [(row.camera, row.image, row.key, row.observed_xy) for row in seed]:
            return None
        return seed, final

    def _residual_copy(self) -> str:
        result = self._derivation
        if isinstance(result, DerivationUnavailable):
            return f"Before/after residuals unavailable: {result.reason}. No comparison is inferred."
        if result.status == "available":
            if self._paired_residuals() is not None:
                return ("Before and after use the same ordered (camera, image, point) observations. "
                        f"The average error was {_number(result.d35_initial_px)} px where the solver began "
                        f"and {_number(result.residual_mean_px)} px after the solve. "
                        "Rows excluded from the solve have no after value; none is assigned zero.")
            return (f"The average error after the solve was {_number(result.residual_mean_px)} px. "
                    f"Same-observation before/after residuals are unavailable: {result.seed_observation_status}. "
                    "Rows excluded from the solve have no after value; they are listed as excluded, not assigned zero.")
        return "Before/after comparison is unavailable for this run."

    def _residual_detail(self) -> str:
        """L2: name the saved diagnostics the plain before/after means reproduce."""
        result = self._derivation
        if isinstance(result, DerivationUnavailable) or result.status != "available":
            return ""
        if self._paired_residuals() is not None:
            return (f"The verified starting mean reproduces D3.5 at {_number(result.d35_initial_px)} px; "
                    f"the saved after mean reproduces D3.6 at {_number(result.residual_mean_px)} px.")
        return f"The saved after mean reproduces D3.6 at {_number(result.residual_mean_px)} px."

    def _residual_rows(self) -> list[tuple[str, str, str, str, str]]:
        result = self._derivation
        if isinstance(result, DerivationUnavailable):
            return [("All cameras", "unavailable", "unavailable", result.reason, "unavailable")]
        observations = result.observations
        rows = []
        camera_names = list(dict(result.per_camera_residual_mean_px))
        counts = {}
        for item in observations:
            counts[item.camera] = counts.get(item.camera, 0) + 1
        seed_counts = result.per_image_observation_count
        paired = self._paired_residuals()
        seed_rows = paired[0] if paired is not None else ()
        for camera_index, (name, after_mean) in enumerate(result.per_camera_residual_mean_px):
            if paired is None:
                before = result.seed_observation_status
            else:
                magnitudes = [row.magnitude_px for row in seed_rows
                              if row.camera == camera_index]
                before = (_number(float(np.mean(magnitudes))) if magnitudes
                          else "no retained rows")
            rows.append((name, str(counts.get(camera_index, 0)), before,
                         _number(after_mean) if after_mean is not None else "no retained rows", "camera mean"))
        if paired is not None and getattr(self, "_learning_level", 1) >= 3:
            # Raw values are L3 (A1 section 1): one row per observation can
            # run to thousands, so L1/L2 keep the per-camera means above.
            # Key each arrow to the exact observation identity. Means remain
            # above; these vectors show direction without implying motion.
            for start, end in zip(paired[0], paired[1]):
                start_xy = ", ".join(_number(value) for value in start.residual_xy)
                end_xy = ", ".join(_number(value) for value in end.residual_xy)
                point = ",".join(str(value) for value in end.key)
                rows.append((f"{end.camera} / image {end.image} / point {point}", "1",
                             _number(start.magnitude_px), _number(end.magnitude_px),
                             f"({start_xy}) → ({end_xy})"))
        excluded = self._excluded_images()
        for index, reason in excluded:
            rows.append((_photo_row(index), "0 retained", "not posed / excluded", f"no after — {reason}", "not applicable"))
        return rows or [("Observations", str(sum(seed_counts)), "unavailable", "unavailable", "unavailable")]

    def _excluded_images(self) -> list[tuple[int, str]]:
        """Use only persisted pose masks whose counts match saved D3.1/D3.2."""
        path = self.run_context.artifact("phase3")
        if path is None or not path.is_file():
            return []
        try:
            from pyCamSet.utils.saving import load_CameraSet
            handler = load_CameraSet(path).calibration_handler
            diagnostics = self.metadata.get("diagnostics") or {}
            expected_before = int(diagnostics["D3.1_n_missing_poses"])
            expected_removed = int(diagnostics["D3.2_n_outlier_removed"])
            if (getattr(handler, "missing_poses_before_outlier_rejection", None) is None
                    and getattr(handler, "missing_poses", None) is not None):
                # A camset read back from disk keeps only the final missing-pose mask.
                # (On a real telecentric rig, all 17 'not posed' rows were lost.)
                final = np.asarray(handler.missing_poses, dtype=bool).reshape(-1)
                if final.ndim != 1 or int(final.sum()) != expected_before + expected_removed:
                    return []
                # Without the earlier mask the two kinds can be told apart only when one is zero.
                if expected_removed == 0:
                    reason = "not posed"
                elif expected_before == 0:
                    reason = "removed by outlier exclusion"
                else:
                    reason = "not posed or removed by outlier exclusion"
                return [(int(i), reason) for i in np.flatnonzero(final)]
            before = np.asarray(handler.missing_poses_before_outlier_rejection, dtype=bool)
            after = np.asarray(handler.missing_poses_after_outlier_rejection, dtype=bool)
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            return []
        if (before.ndim != 1 or after.shape != before.shape or not len(before)
                or np.any(before & ~after) or int(before.sum()) != expected_before
                or int((after & ~before).sum()) != expected_removed):
            return []
        return ([(int(i), "not posed") for i in np.flatnonzero(before)]
                + [(int(i), "removed by outlier exclusion")
                   for i in np.flatnonzero(after & ~before)])

    def _outlier_setting(self) -> str | None:
        """Read the actual persisted solver option; do not infer from removals."""
        params = self.metadata.get("params") or {}
        opts = params.get("problem_options") or {}
        if not isinstance(opts, dict):
            return None
        value = opts.get("outliers", params.get("outliers"))
        if value is True or value == "y":
            return "on"
        if value is False or value == "n":
            return "off"
        return None

    def _outlier_copy(self) -> str:
        """Explain the recorded setting without inferring removals from a setting alone."""
        mode = self._outlier_setting()
        if mode == "off":
            return "Outlier exclusion was off for this run."
        if mode == "on":
            removed = self._excluded_images()
            n_removed = sum(reason == "removed by outlier exclusion" for _, reason in removed)
            stored = (self.metadata.get("diagnostics") or {}).get("D3.2_n_outlier_removed")
            try:
                recorded_count = int(stored)
            except (TypeError, ValueError):
                return "Outlier exclusion was on, but its removal count is unavailable."
            if recorded_count < 0 or recorded_count != n_removed:
                return (f"Outlier exclusion was on; the run records {recorded_count} removal(s), "
                        "but per-image identities could not be verified.")
            return (f"Outlier exclusion was on for this run; {n_removed} image(s) were recorded as newly excluded. "
                    "An image without a solved pose is labelled not posed, not removed.")
        params = self.metadata.get("params") or {}
        opts = params.get("problem_options") or {}
        if "outliers" in opts or "outliers" in params:
            return "Outlier exclusion setting was recorded but is not a recognised on/off value."
        return "Outlier exclusion setting was not recorded for this run."

    def _playback_copy(self) -> str:
        if self._playback.get("status") == "available":
            payload = self._playback["payload"]
            join = next((record.get("payload", {}).get("events", [])
                         for record in self._trace.get("records", [])
                         if record.get("kind") == "seed_join_order"), [])
            preroll = (" The camera joins in the first stage show how the first guess was assembled; "
                       "they are not solver steps." if join else "")
            return ("Recorded accepted solver steps: they show the estimates changing, not the rig "
                    "physically moving. "
                    "Each step shows the solver's estimates after one accepted update; the first "
                    "position is where the solver began, not an accepted step. The objective is the "
                    f"solver's score ({payload.get('loss', 'linear')}), not pixels; the error curve "
                    "is recalculated in pixels from each recorded state. Between recorded steps the "
                    "3D view is interpolated for smooth viewing; only the recorded states are data."
                    + preroll)
        reason = str(self._playback.get("reason", "steps not recorded for this run"))
        if reason == "steps not recorded for this run":
            return ("Steps were not recorded for this run, so only the start and the result are shown. "
                    "To record steps next time, tick ‘Record solver steps for this run’ "
                    "in this phase's settings before you run it. This run and its calibration are not deficient; "
                    "the steps were simply not requested.")
        return (f"Static before/after only. {reason}. The run's saved start and result are not connected by an invented path. "
                "Run Phase 3 again with ‘Record solver steps for this run’ ticked to create a new record.")

    def _step_rows(self) -> list[tuple[str, str, str]]:
        if self._playback.get("status") != "available":
            return ([('Playback', 'unavailable', str(self._playback.get(
                "reason", "steps not recorded for this run")))]
                    + self._saved_final_geometry_rows())
        payload = self._playback["payload"]
        steps = payload.get("steps", [])
        frames = self._playback.get("frames", [])
        # SCI4-7: the same unit and wording as the stage-1 join table (mm when the size is known).
        mm = self._mm_per_unit()

        def _join_value(spread: Any) -> str:
            if spread is None:
                return _number(spread)
            return (f"{es.format_sig(float(spread) * mm)} mm" if mm
                    else f"{_number(spread)} target units")

        result = [(f"Camera {_one_based(event.get('joining'))} joined via camera "
                   f"{_one_based(event.get('joined_from'))}",
                   _join_value(event.get("spread")), "first-guess assembly — not solver steps")
                  for record in self._trace.get("records", [])
                  if record.get("kind") == "seed_join_order"
                  for event in record.get("payload", {}).get("events", [])]
        previous = None
        for index, step in enumerate(steps):
            if previous is None:
                label = "Starting point (not an accepted step)"
                displacement = "—"
            else:
                label = f"Step {step.get('nit', '?')}"
                displacement = (_TELECENTRIC_STEP if _rotation_only(frames[index])
                                else _number(_largest_camera_step(previous, frames[index])))
            result.append((label, _number(step.get("cost")), displacement))
            previous = frames[index]
        # Add actual projected endpoint geometry, never a linearly invented rig.
        if getattr(self._derivation, "fixed_pose_valid", False) is True and len(frames) >= 2:
            result.extend(_geometry_snapshot_rows(frames[0], "Solver starting point"))
            result.extend(_geometry_snapshot_rows(frames[-1], "Saved final result"))
        else:
            result.append(("Rig / target-pose geometry", "unavailable",
                           "fixed-pose check did not establish a shared world frame"))
        nits = [int(step.get("nit", 0)) for step in steps]
        # The first stored entry is nit=0 (the solver start), so include its
        # gap too; otherwise thinning before the first retained step is hidden.
        missing = [step for left, right in zip(nits, nits[1:])
                   for step in range(left + 1, right)]
        if missing:
            result.append(("Thinning", "steps not kept", ", ".join(map(str, missing))))
        return result

    def _saved_final_geometry_rows(self) -> list[tuple[str, str, str]]:
        """Project the saved result alone when no recorded path is available."""
        if getattr(self._derivation, "fixed_pose_valid", False) is not True:
            return [("Saved final rig / target poses", "unavailable",
                     "fixed-pose check did not establish a shared world frame")]
        artifact = self.run_context.artifact("phase3")
        if artifact is None or not artifact.is_file():
            return [("Saved final rig / target poses", "unavailable",
                     "saved Phase 3 camset is missing")]
        try:
            from pyCamSet.utils.saving import load_CameraSet
            saved = load_CameraSet(artifact)
            primitive = saved.calibration_handler.bundlePrimitive
            values = primitive.return_bundle_primitives(
                np.asarray(saved.calibration_params, dtype=np.float64).copy())
            frame = tuple(np.asarray(value, dtype=np.float64).copy() for value in values)
            return _geometry_snapshot_rows(frame, "Saved final result")
        except (OSError, ValueError, TypeError, AttributeError, IndexError):
            return [("Saved final rig / target poses", "unavailable",
                     "saved geometry could not be projected")]

    def _camera_label(self, index: Any) -> str:
        """Name a camera index by its saved camera name, keeping A1's index convention."""
        result = self._derivation
        names = [name for name, _mean in getattr(result, "per_camera_residual_mean_px", ()) or ()]
        try:
            position = int(index)
        except (TypeError, ValueError):
            return f"camera {index}"
        # Cameras are numbered as their badges are, from 1, in every figure and table.
        if 0 <= position < len(names) and names[position]:
            return f"{names[position]} (camera {position + 1})"
        return f"camera {position + 1}"

    def _decision_copy(self) -> str:
        """Return the full (L3) decision text; the frame shows it level by level."""
        return compose(self._decision_frame_copy(), 3)

    def _decision_frame_copy(self) -> dict[str, str]:
        """Split the agreement answer into plain L1 copy, L2 checks and L3 rule details."""
        result = self._derivation
        if isinstance(result, DerivationUnavailable):
            return {"copy": f"No agreement answer: the camera-to-camera comparison is unavailable ({result.reason})."}
        if not result.fixed_pose_valid:
            disposition = self.metadata.get("diagnostics", {}).get("reference_pose_disposition")
            if disposition == "reference image excluded; world frame not fixed":
                return {"copy": ("No valid reference photo in this run: positions are not tied to a fixed frame (see status). "
                                 "Re-run Phase 3 with an observed, usable reference photo, then run Phase 4 from that result.")}
            return {"copy": ("The reference-photo check failed; positions are not tied to a confirmed frame. "
                             "No camera-agreement answer is given. Re-run Phase 3 with an observed, usable "
                             "reference photo, then run Phase 4 from that result.")}
        mm_known = result.saved_point_units in {"m", "mm"} and result.saved_point_units == result.rebuilt_point_units and "legacy" not in result.unit_disposition
        spreads = result.rig_spreads
        unmeasured = tuple(getattr(result, "rig_unmeasured_cameras", ()) or ())
        if not spreads and unmeasured:
            return {"copy": ("Did the cameras agree on where the target was, across your photos? "
                             "Overall answer: Not clearly. No other camera shares at least two posed photos "
                             f"with {self._camera_label(0)}, so their agreement cannot be measured. Next: take "
                             "photos in which the target is seen by the first camera and the others at once.")}
        if not spreads:
            return {"copy": "Too few cameras for the camera-to-camera comparison."}
        rule = self._position_rule()
        judge = rule["judge"]
        level = rule["level_mm"]
        lvl = f"{level:g} mm" if level is not None else ""
        position_threshold_available = (judge and mm_known and
                                        all(item.position_scatter_mm is not None for item in spreads))
        position_over = position_threshold_available and any(
            item.position_scatter_mm > level for item in spreads)
        rotation_over = any(item.rotation_scatter_deg > ROTATION_LEVEL_DEG for item in spreads)
        position_flagged = any(item.position_photo_outliers for item in spreads)
        rotation_flagged = any(item.rotation_photo_outliers for item in spreads)
        photo_rule_checked = all(item.outlier_rule_status == "applied" for item in spreads)

        # Keep the three evidence levels separate: unknown length units must not
        # turn a square-based spread into a millimetre decision.
        position_outcome = ("not judged" if not judge else
                            "Not clearly" if not position_threshold_available else
                            "No" if position_over else "Yes")
        # A spread below its warning level can still hide one late-trigger
        # image. Keep that distinct from a definite spread-level failure.
        rotation_outcome = ("No" if rotation_over else
                            "Not clearly" if rotation_flagged else "Yes")
        photo_outcome = ("Not clearly" if not photo_rule_checked else
                         "Not clearly" if position_flagged or rotation_flagged else "Yes")
        # L1 is the overall answer: a definite failed check is No, all three
        # evidence checks must be Yes for Yes, and gaps leave it Not clearly.
        outcomes = ((position_outcome,) if judge else ()) + (rotation_outcome, photo_outcome)
        overall_outcome = ("No" if "No" in outcomes else
                           "Yes" if all(outcome == "Yes" for outcome in outcomes) else
                           "Not clearly")
        if unmeasured and overall_outcome == "Yes":
            overall_outcome = "Not clearly"  # a camera that could not be measured is not a Yes
        position_reason = (rule["why"] if not judge else
                           f"length units are unverified or a spread is unavailable, so the {lvl} threshold cannot be applied"
                           if not position_threshold_available else
                           f"at least one camera's spread exceeds {lvl}" if position_over else
                           f"all camera spreads are at or below this view's {lvl} warning level")
        rotation_reason = ("at least one camera's spread exceeds 5°" if rotation_over else
                           "all camera spreads are at or below 5°")
        photo_reason = (
            "insufficient usable-photo evidence: the rule needs at least "
            f"{MIN_OUTLIER_N} finite deviations and is declared for N={OUTLIER_DECLARED_N[0]}–{OUTLIER_DECLARED_N[1]}"
            if not photo_rule_checked else
            f"the per-photo rule flagged {self._flagged_photo_labels(spreads)}; the spread check can dilute an isolated image"
            if position_flagged or rotation_flagged else
            "the per-photo rule was applied and flagged no deviations")
        unit_note = ("The mm position conversion assumes the entered square size. " if mm_known and judge
                     else "" if mm_known else
                     "Position is shown in unit-free square units, not converted to length; "
                     "no mm warning threshold is applied because the saved length unit is unverified. ")
        # L1: the A1 question, the overall answer and one plain line per camera.
        lines = ["Did the cameras agree on where the target was, across your photos? "
                 f"Overall answer: {overall_outcome}.",
                 "This compares each camera's first-guess position for every photo, worked out "
                 "separately before the joint solve; the solved rig has one position per camera. "
                 f"Each camera is compared with {self._camera_label(0)}, the first camera."]
        # Define the levels once, before the per-camera lines use them, so a
        # novice reads what a value below them does and does not show.
        # The levels share their numbers with pyCamSet's own Phase 3 rig check
        # (setup_reports.RigConsistencyReport), but that check applies them to the
        # standard deviation of distance and angle size; this view's RMS spread of
        # camera centres and turns is larger, so the two can disagree. That check
        # also always reads pose translations as metres, so for a millimetre
        # target (flat PuzzleBoard) its logged values are 1000x too large.
        log_units = getattr(result, "rebuilt_point_units", None)
        if log_units == "m":
            log_compare = ("pyCamSet's own Phase 3 rig check uses the same numbers on a "
                           "smaller spread measure, so its log may show no warning when this view answers No.")
            log_expert = ("pyCamSet's own Phase 3 rig check uses the same numbers on the "
                          "standard deviation of the relative translation length and rotation angle size, "
                          "which ignores the direction of the change and is smaller.")
        elif log_units == "mm":
            log_compare = ("pyCamSet's own Phase 3 rig check uses the same numbers but reads this "
                           "target's millimetre coordinates as metres, so its logged position values "
                           "are 1000 times too large and its log can warn even when this view answers Yes.")
            log_expert = ("pyCamSet's own Phase 3 rig check uses the same numbers on the "
                          "standard deviation of the relative translation length and rotation angle size, "
                          "which ignores the direction of the change. It multiplies the translation by "
                          "1000 as if the target coordinates were metres; this target's coordinates are "
                          "millimetres, so its logged position values are 1000 times too large.")
        else:
            log_compare = ("pyCamSet's own Phase 3 rig check uses the same numbers on a "
                           "different spread measure, so its log and this view can disagree.")
            log_expert = ("pyCamSet's own Phase 3 rig check uses the same numbers on the "
                          "standard deviation of the relative translation length and rotation angle size, "
                          "which ignores the direction of the change; it reads the translation as metres, "
                          "and this target's length unit is unverified.")
        levels = (f"{lvl} and 5° levels are this view's coarse warning levels" if position_threshold_available
                  else "5° level is this view's coarse warning level")
        below = "them" if position_threshold_available else "it"
        # SCI3-11: a telecentric run judges turns only; say so instead of naming positions.
        spread_words = "camera positions and turns" if judge else "camera turns"
        lines.append(f"The {levels}, applied to the typical (RMS) "
                     f"spread of {spread_words}. A value below {below} is not proof of a rigid "
                     "rig, and a single odd photo is caught by a separate single-photo check (named below "
                     "if it found one). " + log_compare
                     + ("" if judge else f" Positions are not judged: {rule['why']}."))
        for item in spreads:
            label = self._camera_label(item.camera)
            # Capitalise the generic word only; a saved camera name keeps its case.
            if label.startswith("camera "):
                label = "C" + label[1:]
            # Each line says where the camera sits against the level that
            # decides the answer, so a No always has a visible cause at L1.
            turn_over = item.rotation_scatter_deg > 5.0
            if judge and mm_known and item.position_scatter_mm is not None:
                lines.append(f"{label}: its estimated position relative to camera 1 "
                             f"varied by {_number(item.position_scatter_mm)} mm across photos. "
                             + (f"That is more than {lvl}, this view's warning level."
                                if item.position_scatter_mm > level else
                                f"That is within {lvl}, this view's warning level."))
            elif judge and getattr(item, "position_scatter_squares", None) is not None:
                lines.append(f"{label}: its estimated position relative to camera 1 "
                             f"varied between photos by about {_number(item.position_scatter_squares, 1)} squares.")
            elif not turn_over:
                lines.append(f"{label}: its estimated turn relative to camera 1 varied by "
                             f"about {_number(item.rotation_scatter_deg)}° across photos, within 5°, "
                             "this view's rotation warning level.")
            if turn_over:
                # A1 section 6 rotation wording; shown even next to a position value,
                # because rotation alone can decide a No.
                lines.append(f"{label}: from photo to photo, its estimated turn relative to camera 1 "
                             f"varied by about {_number(item.rotation_scatter_deg)}° (typical deviation "
                             "from the average), more than 5°, this view's rotation warning level.")
        for camera in unmeasured:
            label = self._camera_label(camera)
            if label.startswith("camera "):
                label = "C" + label[1:]
            lines.append(f"{label}: not measured, because it shares fewer than two posed photos with "
                         f"{self._camera_label(0)}; its agreement is unknown.")
        if position_flagged or rotation_flagged:
            lines.append(f"Photos that stand out: {self._flagged_photo_labels(spreads)}. "
                         "Check those photos and the capture setup.")
        # A1 section 1: a No or Not clearly answer says what to inspect next.
        next_steps = []
        if overall_outcome == "No":
            next_steps.append("check the photos of each camera above its level for blur, trigger timing "
                              "or very oblique or distant views, or a lens-model problem; if you retake "
                              "photos, re-run from Phase 1")
        elif overall_outcome == "Not clearly":
            if position_flagged or rotation_flagged:
                next_steps.append("check the photos named above for blur, trigger timing or a very "
                                  "oblique or distant view; if you retake photos, re-run from Phase 1")
            if not photo_rule_checked:
                next_steps.append(f"add usable photos (the single-photo check needs at least "
                                  f"{MIN_OUTLIER_N}) and re-run from Phase 1")
            if judge and not position_threshold_available:
                next_steps.append("compare the position spread with an earlier run of the same rig "
                                  "with the same board at a similar distance")
        guidance = ("These are spreads of estimates, not direct measurements of the rig; they can also "
                    "reflect lens-model error, image order, trigger timing, and oblique or distant photos.")
        if mm_known and judge:  # SCI3-11: a telecentric run shows no mm position value
            guidance += " The mm values assume the square size you entered."
        else:
            guidance += (" This number is never 0: each photo's position is worked out separately, and the "
                         "lens estimates add more. There is no fixed good value. Look instead for one photo "
                         "that stands out, if this page names one, or for a clear change between runs of "
                         "the same rig with the same board at a similar distance.")
        lines.append(guidance)
        if next_steps:
            lines.append("What to do next: " + "; and ".join(next_steps) + ".")
        # L2: the separate checks behind the answer, with their reasons and units.
        detail = (
            "Overall rule: No if any check below is No; Yes only if the position, rotation and "
            "single-photo checks are all Yes; otherwise Not clearly. "
            "Source: the seed per-camera target poses for each photo (from the Phase 2 lens "
            "estimates, before the bundle adjustment), not the solved Phase 3 extrinsics. "
            "Spread check (each camera relative to camera 1): "
            f"position = {position_outcome} ({position_reason}); "
            f"rotation = {rotation_outcome} ({rotation_reason}). "
            f"Single-photo check: {photo_outcome} ({photo_reason}). " + unit_note)
        if not mm_known:
            detail += ("The number depends on the board (square size and number of corners), the lens "
                       "estimates and the camera distance, so compare it only between runs with the same "
                       "board at a similar distance.")
        # L3: the rule parameters and what the warning levels are not.
        expert = (f"Single-photo rule: it checks deviations above the relevant level and "
                  f"{OUTLIER_MULTIPLIER:g}× the median, needs at least {MIN_OUTLIER_N} finite "
                  f"deviations, and is declared for N={OUTLIER_DECLARED_N[0]}–{OUTLIER_DECLARED_N[1]}. "
                  + (f"The {lvl} and 5° levels are this view's warning levels on the RMS spread of each "
                     "camera's centre and turn (geodesic angle from the rotation mean), not what a rigid "
                     f"rig should show; the position level is 10 mm or {POSITION_LEVEL_SHARE:.0%} of the "
                     "target's size, whichever is lower. " if judge else
                     "The 5° level is this view's warning level on the RMS spread of each camera's turn "
                     "(geodesic angle from the rotation mean), not what a rigid rig should show. ")
                  + log_expert)
        return {"copy": "\n".join(lines), "detail": detail.strip(), "expert": expert,
                "answer": overall_outcome}

    def _flagged_photo_labels(self, spreads: Any) -> str:
        """Name an identified photo by ordinal when the file mapping is unknown."""
        flagged = []
        for item in spreads:
            for image in sorted(set(item.position_photo_outliers) | set(item.rotation_photo_outliers)):
                flagged.append(f"{self._camera_label(item.camera)}, {es.photo_label(image)} "
                               "(photo file not checkable)")
        return ", ".join(flagged) if flagged else "no named photo"

    def _spread_rows(self) -> list[tuple[str, str, str, str, str]]:
        result = self._derivation
        if isinstance(result, DerivationUnavailable):
            return [("—", "unavailable", "unavailable", "—", result.reason)]
        rows = []
        for item in result.rig_spreads:
            position = (f"{_number(item.position_scatter_mm)} mm" if item.position_scatter_mm is not None
                        else f"about {_number(item.position_scatter_squares, 1)} squares")
            flags = []
            if item.position_photo_outliers:
                flags.append("position photo rule")
            if item.rotation_photo_outliers:
                flags.append("rotation photo rule")
            rows.append((f"{self._camera_label(item.camera)} relative to camera 1", position,
                         f"{_number(item.rotation_scatter_deg)}°", str(item.usable_image_count),
                         "; ".join(flags) if flags else item.outlier_rule_status))
        return rows

    def _single_face_record(self) -> dict[str, Any] | None:
        """Saved D3.14, when the run opted in or left recoverable single-face images out."""
        record = (self.metadata.get("diagnostics") or {}).get("D3.14_single_face_recovery")
        if not isinstance(record, Mapping):
            return None
        # Only when relevant: a telecentric run that placed or left out single-face photos.
        if record.get("telecentric") is False:
            return None
        if record.get("recovered") or record.get("not_recovered"):
            return record
        return None

    def _single_face_entries(self) -> tuple[list[Mapping], list[Mapping]]:
        record = self._single_face or {}
        recovered = [entry for entry in record.get("recovered") or [] if isinstance(entry, Mapping)]
        left_out = [entry for entry in record.get("not_recovered") or [] if isinstance(entry, Mapping)]
        return recovered, left_out

    def _single_face_copy(self) -> dict[str, str]:
        """L1 plain story and next step, L2 the numbers and D3.14, L3 the method and its source."""
        record = self._single_face or {}
        recovered, left_out = self._single_face_entries()
        # template_handler.SINGLE_FACE_REASON_ONE_CAMERA, matched by prefix to keep the solver unimported.
        one_camera = [entry for entry in left_out
                      if str(entry.get("reason", "")).startswith("seen by one camera only")]
        ambiguity = ("In some photos each camera saw only one face of the target. Through a telecentric lens "
                     "one flat face looks the same tilted towards the camera or away from it, so one camera "
                     "alone cannot tell which. ")
        illustration = ("The first figure is an illustration drawn from geometry, not this run's data: one "
                        "camera sees the same image for both tilts, while a second camera at another angle "
                        "sees two different images. ")
        retake = ("What to do: for the photos left out, take photos at angles where a camera sees two faces "
                  "of the target, or make sure several cameras see the target in the same photo.")
        missing = (self.metadata.get("diagnostics") or {}).get("D3.1_n_missing_poses", "unavailable")
        if not record.get("enabled"):
            recoverable = len(left_out) - len(one_camera)
            copy = (ambiguity + "Recovery of these photos was off for this run, so the "
                    f"{len(left_out)} photo(s) listed below were left out of the solve. ")
            if recoverable:
                copy += (f"{recoverable} of them were seen by two or more cameras and could be placed: tick "
                         "‘Recover images where each camera sees only one cube face (telecentric)’ in the "
                         "Phase 3 settings and run Phase 3 again. ")
            if one_camera:
                copy += retake
            return {"copy": copy.strip(),
                    "detail": (f"Source: D3.14, saved with this run (recovery off; the photos were only "
                               f"counted, not fitted). D3.1 counts {missing} photo(s) without a position, "
                               "these included."),
                    "expert": _SINGLE_FACE_METHOD}
        if record.get("telecentric") is False:
            return {"copy": ("Recovery of single-face photos was on, but it applies to telecentric cameras "
                             "only. This run's cameras are not telecentric, so it had no effect."),
                    "detail": "Source: D3.14, saved with this run; it lists no photos.",
                    "expert": _SINGLE_FACE_METHOD}
        copy = ambiguity + illustration
        if recovered or left_out:
            copy += (f"Recovery was on for this run: {len(recovered)} photo(s) were placed by trying both "
                     "tilts against every camera that saw the photo and keeping the one that fits them all; "
                     f"{len(left_out)} could not be placed and were left out of the solve. ")
        else:
            copy += "Recovery was on for this run, and no photo needed it. "
        if recovered:
            copy += ("The second figure shows, for the photo you choose, that each camera's two tilts fit "
                     "that camera about equally, but with its tilt kept, only the true tilt fits all "
                     "cameras. Fitted freely, a start from the wrong tilt can end at the true one. ")
        if left_out:
            copy += retake
        fits = [float(entry.get("rms_px", float("nan"))) for entry in recovered]
        detail = "Source: D3.14, saved with this run. "
        if fits:
            detail += (f"Recovered fits ranged from {_number(min(fits))} to {_number(max(fits))} px "
                       "(RMS over every camera that saw the photo). ")
        detail += (f"D3.1 counts the photos still without a position after this step: {missing}.")
        return {"copy": copy.strip(), "detail": detail, "expert": _SINGLE_FACE_METHOD}

    def _single_face_rows(self) -> list[tuple[str, str, str, str, str]]:
        level = getattr(self, "_learning_level", 1)
        source = "; saved D3.14" if level >= 2 else ""
        recovered, left_out = self._single_face_entries()
        rows = []
        for entry in recovered:
            rows.append((_photo_row(entry.get('image')), "placed", _number(entry.get("rms_px")),
                         ", ".join(map(str, entry.get("cameras") or [])),
                         "both tilts tried against every camera; the one that fits all was kept" + source))
        for entry in left_out:
            rows.append((_photo_row(entry.get('image')), "left out", "—",
                         ", ".join(map(str, entry.get("cameras") or [])),
                         str(entry.get("reason") or "not recorded") + source))
        return rows or [("—", "no single-face photo", "—", "—", "nothing to recover" + source)]


    # -- stage display ---------------------------------------------------------------

    def _level_changed(self, level: int) -> None:
        """Rebuild the level-dependent tables (codes from L2, raw rows at L3) and redraw."""
        self._learning_level = int(level)
        if not self._ready:
            self._show_stage(self.stage_view.current_stage())
            return
        rebuild = {"First guess and solver start": self._seed_rows,
                   "Same observations before and after": self._residual_rows,
                   _SINGLE_FACE_TITLE: self._single_face_rows}
        for frame in self._evidence:
            if frame["title"] in rebuild:
                frame["rows"] = rebuild[frame["title"]]()
        self._show_stage(self.stage_view.current_stage())

    def _camera_changed(self, _index: int) -> None:
        if getattr(self, "_setting_camera", False):  # a stage chose it, not the learner
            self._fill_photo_picker()
            return
        self._user_camera = True
        self._fill_photo_picker()
        if self._current_key in (STAGE_FIRST, STAGE_SOLVER, STAGE_LENS, STAGE_RESULT):
            self._show_stage(self.stage_view.current_stage())

    def _photo_changed(self, _index: int) -> None:
        if self._filling_photos:
            return
        if self._current_key in (STAGE_FIRST, STAGE_SOLVER, STAGE_LENS):
            self._show_stage(self.stage_view.current_stage())

    def _fill_photo_picker(self) -> None:
        """The selected camera's photos in the solve (Photo n · corners), its typical one first
        selected; keeps the current photo when the new camera saw it too."""
        from pyCamSet.workflow.education_visual_p3 import photo_choices, typical_photo
        combo = self.photo_picker
        visual = self._visual
        camera = self._selected_camera()
        previous = combo.currentData()
        self._filling_photos = True
        try:
            combo.clear()
            if visual is None or camera is None:
                combo.addItem("No photo", None)
                combo.setEnabled(False)
                return
            choices = photo_choices(visual, camera)
            for photo, count in choices.items():
                combo.addItem(es.photo_label(photo), int(photo))
                combo.setItemData(combo.count() - 1,
                                  f"{es.photo_label(photo)}: camera {camera + 1} found "
                                  f"{count} corners", Qt.ItemDataRole.ToolTipRole)
            combo.setEnabled(bool(choices))
            wanted = previous if previous in choices else typical_photo(visual, camera)
            found = combo.findData(wanted) if wanted is not None else -1
            combo.setCurrentIndex(max(found, 0))
        finally:
            self._filling_photos = False

    def _selected_photo(self) -> int | None:
        data = self.photo_picker.currentData()
        return int(data) if data is not None else None

    def _show_stage(self, index: int) -> None:
        if not self._ready:
            self._show_pending(index)
            return
        if not 0 <= index < len(self._stage_keys):
            return
        key = self._stage_keys[index]
        self._current_key = key
        self.stage_view.set_busy(None)
        self.stop_for_context_change()
        has_cameras = self.camera_picker.count() > 0
        self.camera_picker.setVisible(
            has_cameras and key in (STAGE_FIRST, STAGE_SOLVER, STAGE_LENS, STAGE_RESULT))
        self.photo_picker.setVisible(
            self._visual is not None and key in (STAGE_FIRST, STAGE_SOLVER, STAGE_LENS))
        self.single_face_image.setVisible(key == STAGE_SINGLE)
        controls = None
        if key == STAGE_GRAPH:
            controls = self.transport
            self._redraw_graph()
        elif key == STAGE_FIRST:
            self._show_first_guess()
        elif key == STAGE_SINGLE:
            self._show_single_face()
        elif key == STAGE_SOLVER:
            controls = self._show_solver()
        elif key == STAGE_LENS:
            best = None if self._user_camera else self._most_bent_camera()
            if best is not None and best != self._selected_camera():
                found = self.camera_picker.findData(best)
                if found >= 0:
                    self._setting_camera = True
                    try:
                        self.camera_picker.setCurrentIndex(found)
                    finally:
                        self._setting_camera = False
            self._show_lens()
        else:
            self._show_result()
        self.stage_view.set_controls(controls)
        self.stage_view.set_caption(self._caption(key))
        self.stage_view.set_chips(self._chips(key))
        self.stage_view.set_glossary(self._glossary(key))
        headers, rows = self._numbers(key)
        self.stage_view.set_numbers(headers, rows)

    # -- shared data for the pictures ---------------------------------------------------

    def _motion(self, factor: float) -> dict[str, np.ndarray]:
        from pyCamSet.workflow.education_visual_p3 import rig_motion
        key = float(factor)
        if key not in self._motion_cache:
            self._motion_cache[key] = rig_motion(self._visual, key)
        return self._motion_cache[key]

    def _motion_factor(self) -> int:
        if self._factor is None:
            from pyCamSet.workflow.education_visual_p3 import motion_factor
            try:
                self._factor = motion_factor(self._visual) if self._recorded() else 1
            except Exception:
                self._factor = 1
        return self._factor

    def _photo_image(self, camera: int, photo: int) -> np.ndarray | None:
        """The photo itself, only when Phase 1's manifest verifies the listing (else None)."""
        key = (int(camera), int(photo))
        if key not in self._photos:
            try:
                from pyCamSet.workflow.education_visual_p2 import load_photo
                from pyCamSet.workflow.education_visual_p3 import photo_source
                if self._photo_source is None:
                    self._photo_source = photo_source(self.run_context)
                name = self._visual.camera_names[int(camera)]
                self._photos[key] = load_photo(self._photo_source.path(name, int(photo)))
            except Exception:
                self._photos[key] = None
        return self._photos[key]

    def _photo_note(self) -> str:
        source = self._photo_source
        if source is not None and getattr(source, "available", False):
            return ""
        # The reason sits in the empty photo area (_photo_missing_word), not in the title, which
        # is too narrow for it (ED3-15).
        return " · no photo"

    def _photo_missing_word(self) -> str:
        """Why no photo is drawn, in a few words for the photo panel's blank area (ED3-15)."""
        source = self._photo_source
        status = str(getattr(source, "status", "") or "") if source is not None else ""
        if status.startswith("could not"):
            return "photo not checked"
        return "photo not verified" if status else "no photo listed"

    def _saved_camsets(self) -> tuple[Any, Any] | None:
        """(Phase 2 input camset, Phase 3 result camset), both from this run's own record;
        only when the derivation verified the Phase 2 input (else None)."""
        if self._camsets is None:
            self._camsets = False
            if not isinstance(self._derivation, DerivationUnavailable):
                try:
                    from pyCamSet.workflow.education_derive import _load_camset
                    used = (self.metadata.get("artifacts") or {}).get("phase2_initial_camset_used")
                    artifact = self.run_context.artifact("phase3")
                    if used and artifact is not None and Path(artifact).is_file():
                        self._camsets = (_load_camset(Path(str(used))), _load_camset(artifact))
                except Exception:
                    self._camsets = False
        return self._camsets or None

    def _lens_changes(self) -> list:
        if self._lens is None:
            self._lens = []
            sets = self._saved_camsets()
            if sets is not None and self._visual is not None:
                from pyCamSet.workflow.education_visual_p3 import lens_changes
                try:
                    self._lens = lens_changes(sets[0], sets[1], self._visual.image_sizes)
                except Exception:
                    self._lens = []
        return self._lens

    def _seed(self) -> tuple | None:
        """The first guess re-run read-only from the run's inputs (cached; several pictures
        read it: the agreement ghosts, the join criterion, the single-face tilts)."""
        if getattr(self, "_seed_cache", None) is None:
            self._seed_cache = False
            sets = self._saved_camsets()
            if sets is not None:
                try:
                    from pyCamSet.workflow.education_visual_p3 import full_seed
                    used = (self.metadata.get("artifacts") or {}).get("phase2_initial_camset_used")
                    self._seed_cache = tuple(full_seed(
                        sets[1], sets[0], phase3_path=self.run_context.artifact("phase3"),
                        phase2_path=Path(str(used)) if used else None))
                except Exception:
                    self._seed_cache = False
        return self._seed_cache or None

    def _join_candidates(self) -> Any:
        """Every link open at each join and the one taken (ED2-11); None unless the recomputed
        links reproduce the recorded join order and scores."""
        if getattr(self, "_join_cache", None) is None:
            self._join_cache = False
            seed = self._seed()
            sets = self._saved_camsets()
            if seed is not None and sets is not None and self._join_events:
                try:
                    from pyCamSet.optimisation.template_handler import target_radius
                    from pyCamSet.workflow.education_visual_p3 import join_candidates
                    found = join_candidates(seed[0], target_radius(
                        sets[1].calibration_handler.target), self._join_events)
                    self._join_cache = found if found is not None and found.reproduced \
                        else False
                except Exception:
                    self._join_cache = False
        return self._join_cache or None

    def _agreement(self) -> Any:
        if self._agreement_cache is None:
            self._agreement_cache = False
            sets = self._saved_camsets()
            seed = self._seed()
            if sets is not None and self._visual is not None and seed is not None:
                from pyCamSet.workflow.education_visual_p3 import camera_agreement
                try:
                    tic, ext, poses = (np.asarray(v, dtype=float) for v in seed[:3])
                    self._agreement_cache = camera_agreement(
                        tic, ext, poses, self._visual.template,
                        telecentric=self._visual.telecentric) or False
                except Exception:
                    self._agreement_cache = False
        return self._agreement_cache or None

    def _phase4_run(self) -> Mapping | None:
        """The latest complete Phase 4 run that continued from this one (or None)."""
        if not hasattr(self, "_phase4_cache"):
            self._phase4_cache = None
            manager = getattr(self.run_context, "manager", None)
            try:
                runs = [run for run in manager.load_runs("phase4")
                        if ((run.get("inputs") or {}).get("phase3_run_id") == self.context.run_id
                            and run.get("status") == "complete" and not run.get("error"))]
                self._phase4_cache = (max(runs, key=lambda run: str(run.get("run_id", "")))
                                      if runs else None)
            except Exception:
                self._phase4_cache = None
        return self._phase4_cache

    def _ladder(self) -> Any:
        if self._ladder_cache is None:
            from pyCamSet.workflow.education_visual_p3 import error_ladder, phase4_camera_rms
            phase2 = None
            manager = getattr(self.run_context, "manager", None)
            try:
                from pyCamSet.workflow.education_trace import resolve_exact_parent
                phase2 = resolve_exact_parent(manager, self.metadata, "phase2")
            except Exception:
                phase2 = None
            phase4 = self._phase4_run()
            try:
                self._ladder_cache = error_ladder(self._visual, phase2, phase4,
                                                  phase4_camera_rms(phase4))
            except Exception:
                self._ladder_cache = False
        return self._ladder_cache or None

    def _lens_effect(self) -> Any:
        """What the lens bending does to the corners (cached; None when it cannot be
        evaluated on this run's replay handler)."""
        if getattr(self, "_lens_effect_cache", None) is None:
            self._lens_effect_cache = False
            if self._replay_handler is not None and self._final_params is not None:
                try:
                    from pyCamSet.workflow.education_visual_p3 import lens_bending_effect
                    effect = lens_bending_effect(self._replay_handler, self._final_params)
                    # The 'on' state must be the solve's own final error, or nothing is drawn.
                    if np.allclose(np.linalg.norm(effect.on_xy, axis=1),
                                   self._visual.states[-1].row_error_px, rtol=1e-6, atol=1e-9):
                        self._lens_effect_cache = effect
                except Exception:
                    self._lens_effect_cache = False
        return self._lens_effect_cache or None

    def _later_lens(self) -> Any:
        if getattr(self, "_later_lens_cache", None) is None:
            from pyCamSet.workflow.education_visual_p3 import later_lens
            self._later_lens_cache = (later_lens(self._phase4_run(), self._visual,
                                                 self._mm_per_unit()) or False)
        return self._later_lens_cache or None

    def _most_bent_camera(self) -> int | None:
        """ED2-2: the camera whose corners the lens bending moved most (the clearest one)."""
        effect = self._lens_effect()
        if effect is not None:
            moved = effect.camera_largest_move()
            if moved:
                return int(max(moved, key=moved.get))
        changes = self._lens_changes()
        return int(max(changes, key=lambda c: c.distortion_shift_px).camera) if changes else None

    def _legacy_start(self) -> bool:
        """An old telecentric run whose solver start dropped the first guess's sideways offsets
        (since fixed in the library): the two start errors then differ (SCI-7)."""
        result = self._derivation
        return (not isinstance(result, DerivationUnavailable)
                and result.weighted_seed_matches_d35 is False
                and self._visual is not None and self._visual.telecentric
                and (result.d35_initial_px or 0) > (result.seed_weighted_mean_px or 0))

    # -- 1 who saw what --------------------------------------------------------------------

    def _anchor(self) -> int:
        if self._join_events:
            return int(self._join_events[0].get("joined_from", 0))
        options = (self.metadata.get("params") or {}).get("problem_options") or {}
        try:
            return int(options.get("ref_cam", 0)) if isinstance(options, Mapping) else 0
        except (TypeError, ValueError):
            return 0

    def _describe_join(self, position: int) -> str:
        if position <= 0:
            return (f"Start: camera {self._anchor() + 1} alone, the camera that shares photos "
                    "with the most others.")
        event = self._join_events[position - 1]
        shared = event.get("shared_views")
        through = (f": the two cameras both placed the target on their own in {int(shared)} "
                   "photos" if shared is not None else "")
        return (f"{_ordinal(position)} join: camera {int(event.get('joining', 0)) + 1} joins via "
                f"camera {int(event.get('joined_from', 0)) + 1}{through}.")

    def _short_join(self, position: int) -> str:
        n = len(self._join_events)
        return "start" if position <= 0 else f"join {position} of {n}"

    def _redraw_graph(self) -> None:
        from pyCamSet.workflow.education_visual_p3 import graph_layout
        visual = self._visual
        level = self._learning_level
        events = self._join_events
        upto = self.transport.position() if self.transport is not None else 0
        n_cams = visual.n_cameras if visual is not None else len(self._camera_names())
        excluded = sorted(index for index, _ in self._excluded_images())
        reason = self._visual_reason

        def drawer(fig, _theme):
            if n_cams == 0:
                ax = fig.add_subplot()
                es.empty_panel(ax, f"Which camera saw which photo is unavailable: {reason}.")
                return
            shared = visual.shared_photos() if visual is not None else None
            positions = graph_layout(visual, n_cams)
            matrix = level >= 2 and shared is not None
            raster = level >= 3 and visual is not None
            if raster:
                # UX2-10: the graph keeps the full upper-left width; the raster sits under
                # A and B only, so the matrix C takes the right column top to bottom.
                grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.25, 1.0],
                                        height_ratios=[1.5, 1.0])
                axes = {"graph": fig.add_subplot(grid[0, 0]),
                        "joins": fig.add_subplot(grid[0, 1]),
                        "matrix": fig.add_subplot(grid[:, 2]),
                        "raster": fig.add_subplot(grid[1, 0:2])}
            else:
                widths = [1.35, 1.25] + ([1.0] if matrix else [])
                grid = fig.add_gridspec(1, len(widths), width_ratios=widths)
                axes = {"graph": fig.add_subplot(grid[0, 0]),
                        "joins": fig.add_subplot(grid[0, 1])}
                if matrix:
                    axes["matrix"] = fig.add_subplot(grid[0, 2])
            letters = iter("ABCD")
            draw_pose_graph(axes["graph"], n_cams, positions, shared, events, upto, level,
                            anchor=self._anchor())
            es.panel_title(axes["graph"], next(letters), "Who saw what",
                           "seen from above · grey: photos shared"
                           + (" · arrow: a join" if events else ""))
            if events:
                candidates = self._join_candidates()
                draw_join_order(axes["joins"], events, upto, self._anchor(), n_cams,
                                badge_size=9 if raster else 11, candidates=candidates,
                                unit_scale=self._mm_per_unit())
                es.panel_title(axes["joins"], next(letters), "How the first guess joined them",
                               "each join takes the best-agreed open link (grey: not taken)"
                               if candidates is not None and candidates.reproduced else
                               "each join takes the best-agreed link")
            else:
                es.empty_panel(axes["joins"], "The join order was not recorded for this run")
                es.panel_title(axes["joins"], next(letters), "How the first guess joined them")
            if matrix:
                draw_shared_matrix(axes["matrix"], shared)
                es.panel_title(axes["matrix"], next(letters), "Photos seen by both",
                               "in the solve · diagonal: photos per camera")
            if raster:
                draw_coverage_raster(axes["raster"], visual.camera_image_counts(), excluded)
                es.panel_title(axes["raster"], next(letters), "Corners per camera and photo",
                               "stronger = more corners · blank: not seen"
                               + (" · ✕ left out" if excluded else ""))

        self.stage_view.draw_figure(drawer)

    # -- 3D stages ---------------------------------------------------------------------

    def _ensure_split(self) -> _SceneAndFigure:
        if self._split is None:
            from pyCamSet.gui.education_scene3d import Scene3D
            self._scene = Scene3D(self)
            self._split = _SceneAndFigure(self._scene, self)
        return self._split

    def _scene_subtitle(self, what: str) -> str:
        """The 3D heading's subtitle (UX2-23: the telecentric drawing note is a tooltip, see
        :meth:`_set_heading`, so the subtitle keeps its key fact)."""
        return what

    def _rig_subtitle(self, lead: str, photo: int | None, level: int) -> str:
        """The 3D heading's subtitle (ED3-5): which photo's pose the dotted target is in, and
        from L2 the chosen photo's outline (the two cubes are never left unnamed)."""
        visual = self._visual
        text = f"{lead} · dots: {es.photo_label(visual.fixed_pose)} (pose fixed in the solve)"
        if level >= 2 and photo is not None and photo != visual.fixed_pose:
            text += f" · accent outline: {es.photo_label(photo)}"
        return text

    def _set_heading(self, letter: str, title: str, subtitle: str) -> None:
        split = self._ensure_split()
        split.heading.set_heading(letter, title, subtitle)
        if self._visual is not None and self._visual.telecentric:
            split.heading.setToolTip(
                f"{split.heading.toolTip()}\nTelecentric cameras see no depth, so each is "
                "drawn at a fixed distance along its viewing direction.")

    def _figure_only(self, message: str) -> None:
        self.stage_view.draw_figure(lambda fig, _t: es.empty_panel(fig.add_subplot(), message))

    def _photo_axes(self, ax: Any, state: float, camera: int, photo: int | None, theme: str,
                    *, crop_states: tuple[int, ...] = ()) -> dict | None:
        """Draw the camera's photo panel at ``state``; returns its artists (or None)."""
        from pyCamSet.workflow.education_visual_p3 import photo_crop, photo_points, photo_rows
        visual = self._visual
        if photo is None:
            es.empty_panel(ax, "This camera saw no photo in the solve")
            return None
        rows = photo_rows(visual, camera, photo)
        observed, predicted, errors = photo_points(visual, state, rows)
        extra = [photo_points(visual, s, rows)[1] for s in crop_states]
        size = visual.image_sizes[camera] if all(visual.image_sizes[camera]) else (
            int(np.ceil(observed[:, 0].max() + 20)), int(np.ceil(observed[:, 1].max() + 20)))
        image = self._photo_image(camera, photo)
        if image is not None:
            size = (image.shape[1], image.shape[0])
        start = photo_points(visual, 0, rows)[2]
        crop = photo_crop(observed, [], size)
        magnify = photo_magnification(max(abs(crop[1] - crop[0]), abs(crop[2] - crop[3])),
                                      float(np.mean(start)) if len(start) else np.nan)
        shown = [observed + magnify * (p - observed) for p in [predicted] + extra]
        crop = photo_crop(observed, shown, size)
        artists = draw_photo_panel(ax, image, observed, shown[0], errors, crop, camera=camera,
                                   size=size, scale=es.error_scale(theme))
        if image is None:
            ax.text(0.5, 0.02, self._photo_missing_word(), transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=8, color=_tokens(ax)["muted"], zorder=7)
        artists["rows"] = rows
        artists["observed"] = observed
        artists["magnify"] = magnify
        self._photo_magnify = magnify
        return artists

    def _show_first_guess(self) -> None:
        visual = self._visual
        if visual is None:
            self._draw_first_guess_without_geometry()
            return
        split = self._ensure_split()
        level = self._learning_level
        split.set_ratio(1, 1)
        camera = self._selected_camera()
        photo = self._selected_photo()
        theme = self.stage_view.theme()
        build_rig_scene(self._scene, visual, self._motion(1.0), state=0, level=level,
                        camera=camera, photo=photo, theme=theme)
        self._set_heading("A", "Where the solver began", self._rig_subtitle(
            "colour: each corner's mean error", photo, level))
        self.stage_view.set_visual(split)
        result = self._derivation
        excluded = dict(self._excluded_images())
        legacy = self._legacy_start()

        def drawer(fig, theme_name):
            rows = [2.0, 1.0] + ([0.8] if level >= 2 and legacy else []) + \
                ([0.8] if level >= 3 else [])
            grid = fig.add_gridspec(len(rows), 1, height_ratios=rows)
            ax = fig.add_subplot(grid[0])
            self._photo_axes(ax, 0, camera, photo, theme_name)
            scale = es.error_scale(theme_name)
            enlarged = getattr(self, "_photo_magnify", 1)
            err = (visual.states[0].row_error_px[
                (visual.rows_camera == camera) & (visual.rows_image == photo)]
                if photo is not None else np.array([]))
            es.panel_title(ax, "B", f"Camera {camera + 1}, {es.photo_label(photo)}"
                           if photo is not None else f"Camera {camera + 1}",
                           "ring: found · dot: predicted"
                           + (f" (gap ×{enlarged})" if enlarged > 1 else "") + " · "
                           f"{es.format_px(float(err.mean())) if err.size else '–'}"
                           + self._photo_note())
            ax2 = fig.add_subplot(grid[1])
            start_errors = visual.per_camera_image_error(0)
            strip_scale = scale
            top = max((float(np.nanmax(v)) for v in start_errors.values() if len(v)),
                      default=0.0)
            if top > scale.hi:  # a rough start: widen the log axis instead of ▲ labels
                strip_scale = es.ErrorScale(theme_name, hi=1000.0)
                strip_scale.ticks = tuple(es.ERROR_TICKS_PX) + (300.0, 1000.0)
            es.strip_plot_by_camera(ax2, start_errors, highlight=camera,
                                    ylabel=es.error_label(per="photo", short=True, unit=False),
                                    xlabel=None, log=strip_scale)
            es.panel_title(ax2, "C", "Every photo at the start",
                           "one dot per photo and camera · log scale")
            letters = iter("DE")
            k = 2
            if level >= 2 and legacy:
                ax3 = fig.add_subplot(grid[k])
                k += 1
                self._draw_seed_per_photo(ax3, result, excluded, visual.n_images,
                                          letter=next(letters))
            if level >= 3:
                ax4 = fig.add_subplot(grid[k])
                errors = visual.states[0].row_error_px
                positive = errors[errors > 0]
                t = _tokens(ax4)
                bins = np.logspace(np.log10(max(positive.min(), 1e-2)),
                                   np.log10(positive.max()), 36)
                counts, edges, patches = ax4.hist(positive, bins=bins, zorder=3)
                centres = np.sqrt(edges[:-1] * edges[1:])
                for patch, colour in zip(patches, scale.colours(centres)):
                    patch.set_facecolor(colour)
                ax4.set_xscale("log")
                from matplotlib.ticker import FuncFormatter
                ax4.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
                ax4.axvline(visual.states[0].mean_px, color=t["text"], linestyle=(0, (4, 2)),
                            linewidth=1.3)
                ax4.set_xlabel("error of one observed corner (px, log)")
                ax4.set_ylabel("corners")
                es.integer_axis(ax4, "y")
                es.panel_title(ax4, next(letters), "Every observation at the start",
                               f"{es.format_count(len(errors))} corners · dashed: mean "
                               f"{es.format_px(visual.states[0].mean_px)}")

        split.pane.draw(drawer)

    def _draw_seed_per_photo(self, ax: Any, result: Any, excluded: Mapping[int, str],
                             n_images: int, letter: str = "C") -> None:
        t = _tokens(ax)
        if isinstance(result, DerivationUnavailable) or not getattr(
                result, "seed_mean_by_image_px", None):
            es.empty_panel(ax, "The first guess's own per-photo error is unavailable"
                           + (f": {result.reason}" if isinstance(result, DerivationUnavailable)
                              else ""))
            return
        values = np.array([np.nan if v is None else float(v)
                           for v in result.seed_mean_by_image_px])
        x = np.arange(len(values))
        finite = np.isfinite(values)
        cap = _nice(float(np.nanpercentile(values[finite], 97))) if finite.any() else 1.0
        ax.vlines(x[finite] + 1, 0, np.minimum(values[finite], cap), color=t["accent"],
                  linewidth=1.6, zorder=3)
        over = finite & (values > cap)
        if over.any():
            ax.scatter(x[over] + 1, np.full(over.sum(), cap), marker="^", color=t["accent"],
                       s=24, zorder=4)
        lost = [i + 1 for i in range(len(values)) if not finite[i]]
        if lost:
            ax.scatter(lost, np.full(len(lost), cap * 0.04), marker="x", color=t["bad"], s=40,
                       zorder=4, label="could not be placed")
            ax.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.0, 1.0),
                      borderaxespad=0.2)
        ax.set_xlim(0, max(len(values), n_images) + 1)
        ax.set_ylim(0, cap * 1.08)
        ax.set_xlabel("Photo")
        ax.set_ylabel(es.error_label(short=True))
        ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
        subtitle = (f"average {es.format_px(result.seed_weighted_mean_px)} over the photos"
                    if self._legacy_start() else
                    f"average over the cameras · {len(lost)} photos could not be placed")
        es.panel_title(ax, letter, "The first guess's own score, per photo", subtitle)

    def _draw_first_guess_without_geometry(self) -> None:
        result = self._derivation
        excluded = dict(self._excluded_images())
        reason = self._visual_reason

        def drawer(fig, _theme):
            ax = fig.add_subplot()
            if isinstance(result, DerivationUnavailable):
                es.empty_panel(ax, f"The first guess cannot be drawn: {reason}; "
                                   f"its numbers are {result.reason}.")
                return
            self._draw_seed_per_photo(ax, result, excluded, len(result.seed_mean_by_image_px),
                                      letter="A")

        self.stage_view.draw_figure(drawer)

    # -- 4 watching the solver ---------------------------------------------------------

    def _show_solver(self):
        visual = self._visual
        if visual is None:
            self._figure_only(f"No pictures of the solve: {self._visual_reason}.")
            return None
        split = self._ensure_split()
        level = self._learning_level
        camera = self._selected_camera()
        photo = self._selected_photo()
        recorded = self._recorded()
        theme = self.stage_view.theme()
        split.set_ratio(1, 1) if level >= 3 else split.set_ratio(11, 10)
        position = self._player.position if (self._player is not None and recorded) else 0.0
        state = int(round(position)) if recorded else len(visual.states) - 1
        factor = self._motion_factor()
        motion = self._motion(factor)
        built = build_rig_scene(self._scene, visual, motion, state=state, level=level,
                                camera=camera, photo=photo, theme=theme, trails=recorded)
        changes = (f"moves shown ×{factor}" if factor > 1 else "moves to scale")
        from pyCamSet.workflow.education_visual_p3 import capped_turns
        capped = capped_turns(visual, factor) if recorded else 0
        if capped:  # SCI2-20: an exaggerated turn that reached the 75° cap is said so
            changes += f" ({capped} turns capped)"
        self._set_heading(
            "A", "The solver at work" if recorded else "The saved result",
            self._rig_subtitle(f"{changes} · trail: each camera's path", photo, level)
            if recorded else "steps were not recorded for this run")
        self.stage_view.set_visual(split)
        if not recorded:
            split.pane.draw(self._draw_solver_unrecorded)
            return None
        frames = rig_frames(visual, motion, ghost=built["ghost"])
        if self._player is None:
            from pyCamSet.gui.education_scene3d import ScenePlayer, pace_weights
            nits = [s.nit for s in visual.states]
            self._player = ScenePlayer(
                self._scene, frames, step_numbers=nits[1:], total_steps=nits[-1],
                step_weights=pace_weights([s.mean_px for s in visual.states]), parent=self)
            self._player.frameShown.connect(self._on_solver_frame)
            self._player.stepReached.connect(self._on_solver_step)
        else:
            self._player.scene = self._scene
            self._player.frames = [dict(frame) for frame in frames]
        self._draw_solver_pane(int(round(position)))
        self._player.show_position(position)
        return self._player

    def _solver_state(self) -> int:
        if self._player is None:
            return 0
        return int(np.clip(np.floor(self._player.position + 1e-9), 0,
                           len(self._visual.states) - 1))

    def _on_solver_step(self, index: int) -> None:
        if self._current_key == STAGE_SOLVER and self._learning_level >= 2:
            self._draw_solver_pane(int(index))

    def _on_solver_frame(self, position: float, _frame: Any) -> None:
        if self._current_key != STAGE_SOLVER:
            return
        self._grow_trails(position, render=False)  # the player renders once, after this
        if not self._solver_artists:
            return
        wait = CURSOR_INTERVAL_MS - self._cursor_clock.elapsed()
        if abs(position - round(position)) > 1e-9 and wait > 0:
            # P3-1: defer, do not drop: the newest skipped position is drawn when the
            # interval has passed (a trailing throttle), so a half-way frame always lands.
            self._pending_position = float(position)
            if not self._cursor_timer.isActive():
                self._cursor_timer.start(int(wait) + 1)
            return
        self._cursor_timer.stop()
        self._pending_position = None
        self._draw_position(position)

    def _draw_pending_position(self) -> None:
        position, self._pending_position = self._pending_position, None
        if position is not None and self._current_key == STAGE_SOLVER and self._solver_artists:
            self._draw_position(position)

    def _draw_position(self, position: float) -> None:
        """Move B's dots, C's cursor, D's bars and the titles to a (fractional) position."""
        self._cursor_clock.restart()
        from pyCamSet.workflow.education_visual_p3 import photo_points
        artists = self._solver_artists
        visual = self._visual
        try:
            if artists.get("bars") is not None:
                artists["bars"](position)
            x, y = artists["curve"]["x"], artists["curve"]["y"]
            px, py = _curve_point(x, y, position)
            artists["curve"]["line"].set_xdata([px, px])
            artists["curve"]["dot"].set_offsets([[px, py]])
            if "inset_dot" in artists["curve"]:
                artists["curve"]["inset_dot"].set_offsets([[px, py]])
                artists["curve"]["inset_dot"].set_visible(position >= 1.0 - 1e-9)
            photo = artists.get("photo")
            if photo is not None:
                observed, predicted, errors = photo_points(visual, position, photo["rows"])
                update_photo_panel(photo, observed, predicted, errors,
                                   es.error_scale(self.stage_view.theme()))
            if artists.get("title") is not None:
                artists["title"](position)
            self._split.pane.canvas.draw_idle()
        except (KeyError, AttributeError, ValueError, TypeError):
            pass

    def _grow_trails(self, position: float, *, render: bool = True) -> None:
        """Each camera's trail up to the playback position (UX2-6: the path grows as the
        solver plays, its arrowhead at the camera). ``render=False`` when the caller renders
        the scene itself (the playback does, once per frame)."""
        if self._scene is None or not self._recorded():
            return
        motion = self._motion(self._motion_factor())
        names = set(self._scene.names())
        grown = False
        for index in range(self._visual.n_cameras):
            name = f"trail{index}"
            if name in names:
                try:
                    self._scene.update_trail(name, trail_points(motion["C"][:, index], position),
                                             render=False)
                    grown = True
                except Exception:  # noqa: BLE001 - a trail never breaks playback
                    pass
        if grown and render:
            # (Scene3D is a QWidget, so its own .render is QWidget.render: draw through the
            # scene implementation.)
            impl = getattr(self._scene, "_impl", None)
            if impl is not None:
                impl.render()

    def _schur(self, state: int) -> Any:
        if state not in self._schur_cache:
            from pyCamSet.workflow.education_visual_p3 import reduced_camera_system
            visual = self._visual
            x = (self._playback.get("arrays") or [None])[state] if self._recorded() else None
            try:
                self._schur_cache[state] = reduced_camera_system(
                    self._replay_handler, x, loss=visual.loss, f_scale=visual.f_scale)
            except Exception as exc:
                self._schur_cache[state] = f"could not be formed ({type(exc).__name__})"
        return self._schur_cache[state]

    def _schur_structure(self) -> Any:
        if self._structure is None:
            from pyCamSet.workflow.education_visual_p3 import schur_structure
            primitive = self._replay_handler.bundlePrimitive
            fixed = (getattr(self._replay_handler, "problem_opts", {}) or {}).get("fixed_pose", 0)
            self._structure = schur_structure(
                self._visual, primitive.n_intr, primitive.n_extr,
                len(self._playback["arrays"][0]), fixed)
        return self._structure

    def _draw_solver_pane(self, state: int) -> None:
        visual = self._visual
        level = self._learning_level
        camera = self._selected_camera()
        photo = self._selected_photo()
        position = self._player.position if self._player is not None else float(state)
        nits = [s.nit for s in visual.states]
        means = [s.mean_px for s in visual.states]
        cam_values = np.array([s.camera_mean_px for s in visual.states])
        finite = cam_values[np.isfinite(cam_values) & (cam_values > 0)]
        limits = ((float(finite.min()) / 1.8, float(finite.max()) * 2.4) if finite.size
                  else (0.1, 100.0))
        state = int(np.clip(state, 0, len(visual.states) - 1))
        n_states = len(visual.states)
        artists: dict[str, Any] = {}

        def drawer(fig, theme_name):
            from pyCamSet.workflow.education_visual_p3 import normal_matrix_blocks, photo_rows
            if level < 3:
                rows = [2.1, 1.0] if level < 2 else [1.9, 1.0, 0.95]
                grid = fig.add_gridspec(len(rows), 1, height_ratios=rows)
                ax = fig.add_subplot(grid[0])
            else:
                grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.1, 1])
                ax = fig.add_subplot(grid[0, 0])
            photo_artists = self._photo_axes(ax, position, camera, photo, theme_name,
                                             crop_states=tuple(range(n_states)))
            rows_here = (photo_rows(visual, camera, photo) if photo is not None
                         else np.zeros(0, dtype=int))

            def photo_title(where: float) -> None:
                err = np.linalg.norm(
                    _interp_residual(visual, where, rows_here), axis=1) if rows_here.size else []
                # L2's B is as narrow as L3's: the short wording keeps the step in the title.
                short = (level >= 2 or bool(self._photo_note())
                         or _step_label(nits, where).startswith("between"))
                label = _step_label(nits, where, short=short)
                magnify = getattr(self, "_photo_magnify", 1)
                # L3's narrow B: between two steps the position is the key fact, so the
                # dots/rings key (said in the caption) gives way to it.
                between = label.startswith("between")
                key = ("" if (short and between) else
                       "dots → rings" if short else "dots slide onto the rings")
                key += ((f" ×{magnify}" if short else f" (gap ×{magnify})")
                        if magnify > 1 else "")
                es.panel_title(ax, "B", (f"Camera {camera + 1}, {es.photo_label(photo)}"
                                         if photo is not None else f"Camera {camera + 1}"),
                               key.strip() + (" · " if key.strip() else "") + f"{label}: "
                               + (es.format_px(float(np.mean(err))) if len(err) else "–")
                               # L3: the photo panel's blank area says why there is no photo
                               + ("" if level >= 3 else self._photo_note()))

            photo_title(position)
            artists["photo"] = photo_artists
            artists["title"] = photo_title
            ax_c = fig.add_subplot(grid[1] if level < 3 else grid[0, 1])
            artists["curve"] = draw_error_curve(ax_c, nits, means, position)
            es.panel_title(ax_c, "C", "Mean error, step by step",
                           f"{len(nits) - 1} recorded steps · log scale")
            if level == 2:
                bars = fig.add_subplot(grid[2])
                patches, values_text = draw_camera_error_bars(
                    bars, interpolated_camera_means(visual.states, position), limits,
                    visual.states[0].camera_mean_px, camera)

                def bars_at(where: float) -> None:
                    # P3-1: D follows the same fractional position as B, C and the 3D.
                    values = interpolated_camera_means(visual.states, where)
                    for patch, value in zip(patches, values):
                        patch.set_height(value if np.isfinite(value) else 0.0)
                    finite = [(c, v) for c, v in enumerate(values) if np.isfinite(v)]
                    for text, (c, value) in zip(values_text, finite):
                        text.xy = (c, value)
                        text.set_text(es.format_number(value))
                    es.panel_title(bars, "D", "Each camera's mean error",
                                   _at_step(nits, where) + " · grey ticks: at the start")

                bars_at(position)
                artists["bars"] = bars_at
            if level >= 3:
                a_h = fig.add_subplot(grid[1, 0])
                try:
                    blocks = normal_matrix_blocks(self._schur_structure())
                    draw_normal_matrix(a_h, blocks)
                    es.panel_title(a_h, "D", "What the solver solves",
                                   f"{blocks['n_kept'] + blocks['n_eliminated']} numbers · "
                                   "photos stand alone")
                except Exception as exc:
                    es.empty_panel(a_h, f"The solver's equations are unavailable "
                                        f"({type(exc).__name__})")
                a_s = fig.add_subplot(grid[1, 1])
                system = self._schur(state)
                if isinstance(system, str):
                    es.empty_panel(a_s, f"The camera system {system}")
                else:
                    from pyCamSet.workflow.education_visual_p3 import coupling_matrix
                    image = draw_coupling(a_s, coupling_matrix(system))
                    # ED3-8: say what a tie is: how much two cameras' numbers depend on each
                    # other through the photos they share (relative, 0..1).
                    bar = _side_label_bar(fig, image, a_s, "shared-photo link strength")
                    bar.set_ticks([0, 1])
                    bar.set_ticklabels(["0 weak", "1 strong"])
                    es.panel_title(a_s, "E", "The camera system",
                                   f"{system['n_kept']} numbers, solved first (shade: link)")

        self._split.pane.draw(drawer)
        self._solver_artists = artists

    def _draw_solver_unrecorded(self, fig, _theme) -> None:
        visual = self._visual
        ax = fig.add_subplot()
        if visual is None or len(visual.states) < 2:
            es.empty_panel(ax, "Steps were not recorded for this run")
            return
        before = {i: v for i, v in enumerate(visual.states[0].camera_mean_px)}
        after = {i: v for i, v in enumerate(visual.states[-1].camera_mean_px)}
        es.paired_bars_by_camera(ax, before, after, labels=("start", "result"), xlabel=None,
                                 ylabel=es.error_label(short=True, unit=False))
        ax.set_yscale("log")
        _plain_log_axis(ax)
        es.panel_title(ax, "B", "Start and result only",
                       "steps were not recorded, so no path between them is drawn")

    # -- 5 lens fine-tuning ------------------------------------------------------------

    def _show_lens(self) -> None:
        """Lens tuning as its EFFECT on this run's corners (SCI2-1 / ED2-2): A the chosen
        photo with arrows for how far the tuned bending (distortion and eps) moved each
        predicted corner, B each camera's error with the bending switched off and on; L2 C
        the same photos in Phase 4 (target refined too) when a Phase 4 run continued from
        this one; L3 D the depth term eps in 1/m, Phase 3 against Phase 4 (as Phase 4 shows
        it). The 'image centre moved' bars are gone: for a telecentric camera that point is the
        principal point, which is also the image of the reference photo's target origin (SCI2-2),
        so its position depends on the frame the run was solved in."""
        changes = self._lens_changes()
        visual = self._visual
        camera = self._selected_camera()
        level = self._learning_level
        sets = self._saved_camsets()
        if not changes or sets is None or camera is None:
            self._figure_only("The lenses before and after Phase 3 cannot be compared for this "
                              "run (its Phase 2 input is not verified).")
            return
        change = next((c for c in changes if c.camera == camera), changes[0])
        photo = self._selected_photo()
        image = self._photo_image(change.camera, photo) if photo is not None else None
        tele = change.telecentric
        effect = self._lens_effect()
        later = self._later_lens()
        unit = self._mm_per_unit()
        from pyCamSet.workflow.education_visual_p3 import eps_per_metre, photo_rows
        eps3 = {c.camera: eps_per_metre(c.eps_after, unit) for c in changes} if tele else {}
        eps3 = {k: v for k, v in eps3.items() if v is not None}
        edge =bool(self._largest_move_place(effect)) if effect is not None else False
        small_gain = False
        if effect is not None:
            try:
                off = float(np.linalg.norm(effect.off_xy, axis=1).mean())
                on = float(np.linalg.norm(effect.on_xy, axis=1).mean())
                moved = np.asarray(effect.moved_px, dtype=float)
                small_gain = (off > 0 and (off - on) < 0.3 * off
                              and float(moved.mean()) < 0.3 * float(moved.max()))
            except Exception:
                small_gain = False

        def drawer(fig, theme):
            has_d = level >= 3 and (not tele or bool(eps3 and later is not None
                                                     and later.eps_per_m))
            panels = ["B"] + (["C"] if level >= 2 else []) + (["D"] if has_d else [])
            grid = fig.add_gridspec(len(panels), 2, width_ratios=[1.15, 1])
            ax = fig.add_subplot(grid[:, 0])
            if effect is not None and photo is not None:
                rows = photo_rows(visual, change.camera, photo)
                others = np.flatnonzero((visual.rows_camera == change.camera)
                                        & (visual.rows_image != photo))
                if len(others) > 500:
                    others = others[np.linspace(0, len(others) - 1, 500).astype(int)]
                # ED3-1: L1 draws only this photo's corners (the grey field of the camera's
                # other photos, much of it where this photo has no target, comes at L2).
                field = ((visual.rows_xy[others], effect.moved_xy[others])
                         if level >= 2 else None)
                factor = draw_lens_effect(ax, image, visual.rows_xy[rows],
                                          effect.moved_xy[rows], change.centre_after,
                                          change.image_size, change.camera, field=field)
                moved = effect.moved_px[rows]
                es.panel_title(ax, "A", f"Camera {change.camera + 1}, {es.photo_label(photo)}",
                               (f"arrow: how far the bending moved a corner"
                                + (f" ×{factor}" if factor > 1 else "")
                                + (" · grey: other photos" if field is not None else
                                   " · this photo only: typical move "
                                   f"{es.format_sig(float(np.mean(moved)))} px"))
                               if moved.size else "no corners in this photo")
            else:
                cam = sets[1][visual.camera_names[change.camera]]
                factor = 1 if change.distortion_shift_px <= 0 else max(1, min(int(_nice(
                    0.03 * max(change.image_size) / change.distortion_shift_px)), 50))
                draw_lens_grid(ax, image, change, cam, factor, change.camera)
                es.panel_title(ax, "A", f"Camera {change.camera + 1}'s lens after Phase 3",
                               f"grid bent by the fitted distortion, ×{factor}"
                               if change.distortion_fitted else "no distortion fitted")
            slots = iter(range(len(panels)))
            b = fig.add_subplot(grid[next(slots), 1])
            if effect is not None:
                draw_bending_effect(b, effect.camera_mean("off"), effect.camera_mean("on"),
                                    change.camera)
                es.panel_title(b, "B", "Each camera's error, bending off → on",
                               "○ switched off, nothing re-fitted · ● as solved")
                # ED3-1: why a 15 px bend buys under 1 px: few corners sit where it is large
                # (said only when it is so: a small gain and a typical move well below the
                # largest).
                if small_gain:
                    b.text(0.01, 0.02, "small gain: most corners sit where the bend is small"
                           + ("; the largest moves are at the picture edge" if edge else ""),
                           transform=b.transAxes, ha="left", va="bottom", fontsize=9,
                           color=_tokens(b)["muted"], wrap=False)
            else:
                es.bars_by_camera(b, {c.camera: c.distortion_shift_px for c in changes},
                                  ylabel="largest shift", unit="px", xlabel=None,
                                  highlight=change.camera, value_labels=False)
                es.panel_title(b, "B", "Lens distortion", "largest shift it makes in the frame")
            if level >= 2:
                c_ax = fig.add_subplot(grid[next(slots), 1])
                if later is not None:
                    draw_later_shift(c_ax, {c.camera: c.distortion_shift_px for c in changes},
                                     later.distortion_shift_px, change.camera)
                    es.panel_title(c_ax, "C", "Same photos, Phase 4 (target refined too)",
                                   "largest distortion shift · ○ Phase 3 · ● Phase 4")
                elif tele and eps3:
                    draw_eps(c_ax, eps3, None, change.camera)
                    es.panel_title(c_ax, "C", "Depth term ε (1/m = % per cm)",
                                   "may also absorb target-shape error")
                else:
                    values = {c.camera: 100 * c.scale_change for c in changes}
                    es.bars_by_camera(c_ax, values, ylabel="change", unit="%", xlabel=None,
                                      highlight=change.camera, value_labels=False)
                    low = min(list(values.values()) + [0.0])
                    high = max(list(values.values()) + [0.0])
                    pad = 0.25 * max(high - low, 1e-6)
                    c_ax.set_ylim(min(low - pad, 0), high + pad)
                    c_ax.axhline(0, color=_tokens(c_ax)["text"], linewidth=0.8)
                    _plain_value_axis(c_ax)
                    es.panel_title(c_ax, "C", "Focal length", "change from Phase 2")
            if has_d:
                d = fig.add_subplot(grid[next(slots), 1])
                if tele and eps3 and later is not None and later.eps_per_m:
                    draw_eps(d, eps3, later.eps_per_m, change.camera)
                    es.panel_title(d, "D", "Depth term ε (1/m = % per cm)",
                                   "○ Phase 3 · ● Phase 4: smaller, other sign")
                elif not tele:
                    es.bars_by_camera(d, {c.camera: c.centre_shift_px for c in changes},
                                      ylabel="moved", unit="px", xlabel=None,
                                      highlight=change.camera, value_labels=False)
                    _plain_value_axis(d)
                    es.panel_title(d, "D", "Principal point", "moved from Phase 2's estimate")
                else:
                    es.empty_panel(d, "No later run to compare ε with")
                    es.panel_title(d, "D", "Depth term ε", "centred on the principal point")
            del theme

        self.stage_view.draw_figure(drawer)

    def _largest_move_place(self, effect: Any) -> str:
        """" at the picture edge" when the most-moved corner lies in the outer 10 % of its
        camera's frame (ED3-1), else "" (never claimed without checking)."""
        visual = self._visual
        try:
            k = int(np.argmax(effect.moved_px))
            cam = int(effect.rows_camera[k])
            x, y = visual.rows_xy[k]
            w, h = visual.image_sizes[cam]
            if w and h and min(x, w - x, y, h - y) <= 0.10 * min(w, h):
                return " at the picture edge"
        except Exception:
            pass
        return ""

    def _mm_per_unit(self) -> float | None:
        try:
            from pyCamSet.utils.target_units import millimetres_per_point_data_unit
            handler = self._replay_handler
            value = millimetres_per_point_data_unit(handler.target) if handler is not None else None
            return float(value) if value else None
        except Exception:
            return None

    # -- 6 result ----------------------------------------------------------------------

    def _show_result(self) -> None:
        visual = self._visual
        if visual is None:
            self._draw_result_without_geometry()
            return
        split = self._ensure_split()
        level = self._learning_level
        split.set_ratio(2, 3)
        means = visual.states[-1].camera_mean_px
        camera = self._selected_camera()
        theme = self.stage_view.theme()
        build_rig_scene(self._scene, visual, self._motion(1.0), state=len(visual.states) - 1,
                        level=level, camera=camera, photo=None, theme=theme,
                        labels=[f"({es.format_px(v)})" for v in means])  # UX6-1: unit inside the bracket
        self._set_heading("A", "The target in the rig, after the solve",
                          "colour: each corner's mean error · in brackets: each camera's mean error")
        self.stage_view.set_visual(split)
        result = self._derivation
        spreads = (getattr(result, "rig_spreads", ()) if not isinstance(
            result, DerivationUnavailable) and getattr(result, "fixed_pose_valid", False) else ())
        mm_known = (not isinstance(result, DerivationUnavailable)
                    and getattr(result, "saved_point_units", None) in {"m", "mm"}
                    and result.saved_point_units == result.rebuilt_point_units
                    and "legacy" not in str(result.unit_disposition))
        ladder = self._ladder()
        agreement = self._agreement()
        unit = self._mm_per_unit()
        factor = (agreement_factor(agreement, _target_geometry(visual)[1])
                  if agreement is not None else 1)
        rule = self._position_rule()
        shown_cams = None
        if agreement is not None and level < 3 and len(agreement.alone) > 1:
            # UX3-15: at L1/L2 the chosen camera alone against the shared pose (all cameras at
            # L3); if it has no own pose of this photo, the one that disagrees most.
            far = sorted(agreement.alone, key=lambda c: -(
                agreement.offset[c] + np.radians(agreement.angle_deg[c])
                * 0.5 * _target_geometry(visual)[1]))
            shown_cams = [camera] if camera in agreement.alone else far[:1]

        def drawer(fig, _theme):
            # L1: B ladder over C | D. L2: B | E on top. L3: the strip D gives way to F (UX-1:
            # a 2x2 grid, every panel at least half the pane wide).
            if level < 2:
                grid = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.4], width_ratios=[1, 1.15])
                cells = {"B": grid[0, :], "C": grid[1, 0], "D": grid[1, 1]}
            else:
                grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], width_ratios=[0.8, 1.2])
                cells = {"B": grid[0, 0], "E": grid[0, 1], "C": grid[1, 0],
                         ("F" if level >= 3 else "D"): grid[1, 1]}
            top = fig.add_subplot(cells["B"])
            if ladder is not None and ladder.phase3_rms:
                draw_error_ladder(top, ladder)
                if ladder.phase2_rms:
                    # SCI3-2: Phase 4 shares one pose per photo too and ends below Phase 2,
                    # so the rise is the drawn cube, not the sharing.
                    refined = bool(getattr(ladder, "phase4_rms", None))
                    es.panel_title(top, "B", "Why the error rose",
                                   ("RMS per camera · alone → one rig on the drawn cube"
                                    + (" → cube refined" if refined else ""))
                                   if level < 2 else "RMS per camera, phase by phase")
                else:  # SCI2-24: no Phase 2 values, so no 'rose' story
                    es.panel_title(top, "B", "Error in one rig",
                                   "RMS per camera · no Phase 2 record")
            else:
                es.empty_panel(top, "The per-camera errors are not available")
            c = fig.add_subplot(cells["C"])
            if agreement is not None:
                draw_agreement(c, agreement, visual.template, visual.up, unit, factor,
                               cameras=shown_cams)
                es.panel_title(c, "C", "Cameras must agree",
                               f"{es.photo_label(agreement.photo)}: "
                               + (f"camera {shown_cams[0] + 1} alone" if shown_cams
                                  and len(shown_cams) == 1 else "each camera alone")
                               + (f" ×{factor}" if factor > 1 else "") + " vs shared")
            else:
                es.empty_panel(c, "Each camera's own pose of a photo is not available")
            if "D" in cells:
                d = fig.add_subplot(cells["D"])
                es.strip_plot_by_camera(d, visual.per_camera_image_error(len(visual.states) - 1),
                                        highlight=camera,
                                        ylabel=es.error_label(per="photo", short=True,
                                                              unit=False),
                                        xlabel=None, log=es.error_scale(d))
                es.panel_title(d, "D", "Every photo, after", "one dot per photo and camera · log")
            if "E" in cells:
                e = fig.add_subplot(cells["E"])
                if spreads:
                    draw_spreads(e, spreads, mm_known, position_level_mm=rule["level_mm"],
                                 judge_position=rule["judge"],
                                 unchecked=[int(s.camera) for s in spreads
                                            if s.outlier_rule_status != "applied"])
                    es.panel_title(e, "E", "Did the cameras agree?",
                                   "turn spread vs camera 1 · position not judged"
                                   if not rule["judge"] else
                                   f"spread vs camera 1 · 100 % = {rule['level_mm']:g} mm or 5°")
                else:
                    es.empty_panel(e, "Camera agreement was not measured for this run")
            if "F" in cells:
                f = fig.add_subplot(cells["F"])
                limit = _nice(float(np.nanpercentile(
                    np.abs(visual.final_residual_xy), 99))) if len(visual.final_residual_xy) else 1
                cells = draw_residual_density(f, visual, limit)
                _side_label_bar(fig, cells, f, "corners per cell")
                es.panel_title(f, "F", "Every remaining error",
                               "shade: corners per cell · rings: 1σ")

        split.pane.draw(drawer)

    def _draw_result_without_geometry(self) -> None:
        result = self._derivation
        reason = self._visual_reason

        def drawer(fig, _theme):
            ax = fig.add_subplot()
            if isinstance(result, DerivationUnavailable) or not result.per_camera_residual_mean_px:
                es.empty_panel(ax, f"The result cannot be drawn: {reason}.")
                return
            values = {i: (np.nan if v is None else float(v))
                      for i, (_n, v) in enumerate(result.per_camera_residual_mean_px)}
            es.bars_by_camera(ax, values, ylabel=es.error_label(short=True, unit=False),
                              unit="px", xlabel=None, value_format="{:.2f}")
            es.panel_title(ax, "A", "Mean error per camera after the solve",
                           "recalculated from the saved result")

        self.stage_view.draw_figure(drawer)

    # -- 3 single-face photos ----------------------------------------------------------

    def _show_single_face(self) -> None:
        level = self._learning_level
        image = self.single_face_image.currentData()
        teaching = (self._single_face or {}).get("teaching") or {}
        candidates = (teaching.get(str(image)) if isinstance(teaching, Mapping)
                      and image is not None else None)
        candidates = [entry for entry in candidates or [] if isinstance(entry, Mapping)]
        recovered, left_out = self._single_face_entries()
        index = self._camera_index()
        self._single_face_candidates = candidates

        demo = self._tilt_demo(image, candidates)
        self._single_face_demo = demo
        photo = (self._photo_image(demo.viewer, demo.image)
                 if demo is not None and self._visual is not None else None)

        def drawer(fig, _theme):
            letters = iter("ABCDE")
            right = 2 if level >= 2 else 1
            widths = [0.85, 1.6] + ([1.35] if level >= 2 else [])
            # UX7-3: at L2 the tilt photo is landscape, so it takes the top of the middle column
            # and the single-face scatter (E) sits under it; D takes the whole right column.
            # The old two-row photo cell left blank bands above and below the picture.
            tall = level >= 2 and demo is not None
            grid = fig.add_gridspec(2, 1 + right, width_ratios=widths,
                                    height_ratios=[1.3, 1] if tall else [1, 1])
            one, two = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
            draw_tilt_ambiguity(one, two)
            es.panel_title(one, next(letters), "One camera", "both tilts, same picture")
            es.panel_title(two, next(letters), "Two cameras", "the tilts now differ")
            handles, labels = two.get_legend_handles_labels()
            two.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                       ncols=1, fontsize=9)
            main = fig.add_subplot(grid[0, 1] if tall else grid[:, 1])
            if demo is not None:
                final = next((float(e.get("rms_px")) for e in recovered
                              if str(e.get("image")) == str(demo.image)
                              and e.get("rms_px") is not None), None)
                draw_tilt_demo(main, photo, demo, self._visual.image_sizes[demo.viewer]
                               if self._visual is not None else (0, 0), final_px=final)
                es.panel_title(main, next(letters),
                               f"{es.photo_label(demo.image)} in camera {demo.viewer + 1}",
                               f"camera {demo.proposer + 1}'s two tilts · ring: found · "
                               "✕: the flipped-board alternative, rejected"
                               + ("" if photo is not None else self._photo_note()))
            else:
                draw_candidate_fits(main, image, candidates, level, index)
                es.panel_title(main, next(letters),
                               f"{es.photo_label(int(image))}: both tilts, every camera"
                               if image is not None else "Candidate fits",
                               "bars: fit error · ✓ the tilt kept")
            if level >= 2:
                b = fig.add_subplot(grid[:, 2] if tall else grid[0, 2])
                if demo is not None:
                    draw_candidate_fits(b, image, candidates, level, index)
                    es.panel_title(b, next(letters), "Both tilts in every camera",
                                   "+/− a camera's two tilts · ✓ kept")
                c = fig.add_subplot(grid[1, 1] if tall else
                                    (grid[1, 2] if demo is not None else grid[:, 2]))
                draw_recovery_overview(c, recovered, left_out, image, level)
                es.panel_title(c, next(letters), "Every single-face photo",
                               f"{len(recovered)} placed · {len(left_out)} left out")

        self.stage_view.draw_figure(drawer)

    def _tilt_demo(self, image: Any, candidates: list) -> Any:
        """The L1 picture's data for one recovered photo (cached per photo; None when it cannot
        be rebuilt from this run's own inputs, then the bar chart is the main figure)."""
        cache = self.__dict__.setdefault("_tilt_cache", {})
        if image is None:
            return None
        key = str(image)
        if key not in cache:
            cache[key] = None
            sets, seed = self._saved_camsets(), self._seed()
            if sets is not None and seed is not None:
                from pyCamSet.workflow.education_visual_p3 import tilt_demo
                demo = tilt_demo(sets[1], sets[0], int(image), candidates, seed=seed)
                # Only a rebuild that reproduces the saved fits is drawn (never a guess).
                cache[key] = demo if demo is not None and demo.matches_saved else None
        return cache[key]

    # -- captions, chips, numbers --------------------------------------------------------

    def _caption(self, key: str) -> str:
        """L1: at most two short sentences; L2 and L3 each add one (additive)."""
        parts = self._caption_parts(key)
        return " ".join(part for part in parts[:self._learning_level] if part).strip()

    def _first_guess_detail(self) -> str:
        """L2: the solver's start in px, and (old runs only) why the first guess's own score
        differs from it (SCI-7)."""
        start = es.format_px(self._visual.states[0].mean_px)
        result = self._derivation
        if isinstance(result, DerivationUnavailable):
            return f"The mean error at this start is {start}."
        if self._legacy_start():
            # SCI2-18: no promise of the old number; a rerun computes its own first guess.
            return (f"Old run: the solver began at {start}; the first guess scores "
                    f"{es.format_px(result.seed_weighted_mean_px)} (D), worked out differently "
                    "and not compared.")
        if result.weighted_seed_matches_d35 is False:
            return (f"The mean error at this start is {start}; the first guess's own score "
                    f"({es.format_px(result.seed_weighted_mean_px)}) is worked out differently "
                    "and is not compared.")
        return (f"A rough start is normal: the solver only needs to begin close enough to "
                f"slide downhill (mean {start}).")

    def _caption_parts(self, key: str) -> list[str]:
        """[L1 text, L2 addition, L3 addition]. UX2-5: written to fit 2 / 3 / 4 rendered lines
        at 1280 px; asides and next steps live in the Numbers drawer and tooltips."""
        visual = self._visual
        if key == STAGE_GRAPH:
            if visual is None and not self._camera_names():
                return [f"Which camera saw which photo cannot be shown: {self._visual_reason}.",
                        "", ""]
            l1 = "Badges are cameras seen from above; grey lines join cameras that saw a photo " \
                 "together (thicker = more)."
            l1 += (" The first guess adds the cameras one at a time, the best-agreed link first "
                   "(B); press Play." if self._join_events else
                   " The order in which the first guess linked them was not recorded.")
            l2 = "Each join places one more camera, using a photo it shares with a camera already placed."
            l3 = "C counts photos both saw; D, the corners each camera saw in each photo."
            return [l1, l2, l3]
        if key == STAGE_FIRST:
            if visual is None:
                return [self._seed_copy(), "", ""]
            l1 = ("The first guess puts every camera and every photo's target in one rig; on "
                  "the photo, a dot is a predicted corner, a ring a found one.")
            l2 = self._first_guess_detail()
            l3 = (f"3D: the target in {es.photo_label(visual.fixed_pose)}'s pose, which the solve "
                  "fixes, removing the rig's overall position and turn; "
                  f"{'E' if self._legacy_start() else 'D'}: the pixel error of every observation.")
            return [l1, l2, l3]
        if key == STAGE_SINGLE:
            recovered, left_out = self._single_face_entries()
            record = self._single_face or {}
            one_camera = [entry for entry in left_out
                          if str(entry.get("reason", "")).startswith("seen by one camera only")]
            retake = "for the photos left out, take photos at angles where a camera sees two faces."
            if not record.get("enabled"):
                recoverable = len(left_out) - len(one_camera)
                l1 = (f"In {len(left_out)} photos every camera saw only one face of the target; "
                      "recovery was off for this run, so they were left out of the solve.")
                if recoverable:
                    label = "‘Recover images where each camera sees only one cube face (telecentric)’"
                    l1 += (f" {recoverable} of them {'was' if recoverable == 1 else 'were'} seen by "
                           f"two or more cameras: tick {label} in the Phase 3 settings and run "
                           "Phase 3 again.")
                else:
                    l1 += " To use them, " + retake
                return [l1, "Source: D3.14, saved with this run (the photos were only counted, not "
                        "fitted).", "The method is in the Numbers drawer."]
            if record.get("telecentric") is False:
                return ["Recovery of single-face photos was on, but it applies to telecentric "
                        "cameras only; this run's cameras are not telecentric, so it had no effect.",
                        "Source: D3.14, saved with this run.", ""]
            demo = getattr(self, "_single_face_demo", None)
            l1 = ("One telecentric camera (parallel rays, so distance does not change its image size) "
                  "cannot tell a face tilted towards it from away (A, B).")
            if demo is not None:
                l1 += (f" C: camera {demo.proposer + 1}'s two tilts in camera "
                       f"{demo.viewer + 1}'s photo; only the kept one (✓) hits the rings.")
            else:
                l1 += " C: both tilts in every camera; the kept one (✓) fits them all."
            overview = "E" if demo is not None else "D"
            l2 = (f"{len(recovered)} placed, {len(left_out)} left out ({overview}); a fit under "
                  "20 px only has to pick the tilt, the solver then refines the rest.")
            l3 = ("To place the rest, take photos where a camera sees two faces; the method is "
                  "in this stage's Numbers drawer." if left_out else
                  "The method and its source are in the Numbers drawer.")
            return [l1, l2, l3]
        if key == STAGE_SOLVER:
            if visual is None or not self._recorded():
                return [self._playback_copy(), "", ""]
            camera = self._selected_camera()
            factor = self._motion_factor()
            l1 = ("Each accepted step nudges every camera and photo pose so the dots slide onto "
                  "the rings (B) and the error falls (C); press Play.")
            if factor > 1:
                l1 += f" 3D moves are shown ×{factor}, with trails."
            # SCI2-16: one meaning per panel letter at each level (D is the bars at L2 only).
            # The 3D outline is named in the heading's subtitle (ED3-5), not here, so L3 stays
            # within four rendered lines.
            if self._learning_level >= 3:
                l2 = ""
            else:
                l2 = (f"D: each camera's mean error (camera {(camera or 0) + 1} ringed).")
            solver_note = ("" if str(visual.solver).lower() in ("schur", "") else
                           f" (This run used {visual.solver}, not the method drawn here.)")
            l3 = ("Each photo's pose is tied only to itself (D), so the solver sets them aside, "
                  "solves for the cameras (E), then works the photo poses out again."
                  + solver_note)
            return [l1, l2, l3]
        if key == STAGE_LENS:
            changes = self._lens_changes()
            tele = bool(changes) and changes[0].telecentric
            fitted = [c for c in changes if c.distortion_fitted]
            before = any(any(c.distortion_before) for c in changes)  # SCI2-17
            # ED3-11: ε is named with what it does, never bare.
            bending = "distortion, and ε: size change with depth" if tele else "distortion"
            if not fitted and not tele:
                l1 = ("Phase 3 refines each lens with the whole rig; no distortion was fitted "
                      "for this run (A, B).")
            elif self._lens_effect() is not None:
                # Two rendered lines at 1280 px at L1 (caption budget, UX3-2).
                l1 = (f"Each lens may also bend ({bending}"
                      + ("" if before else "; Phase 2 had none") + "). A: how far corners "
                      "moved; B: error, bend off and on.")
            else:
                l1 = (f"Phase 3 also lets each lens bend ({bending}); A draws it on a "
                      "grid, B per camera.")
            later = self._later_lens()
            if later is not None and later.distortion_shift_px:
                top = max(later.distortion_shift_px.values())
                l2 = ("Partly it fixes a target not built as drawn; Phase 4 keeps "
                      f"≤ {es.format_sig(top)} px (C).")
            else:
                l2 = ("Part of it can make up for a printed target that is not its drawing; "
                      "Phase 4 refines the target too and separates them.")
            if tele:
                l3 = ("+ is the bend centre: the principal point, the image of the optical axis; "
                      "ε in 1/m = % per cm.")
            else:
                l3 = "Every coefficient, before and after, is listed in the Numbers drawer."
            return [l1, l2, l3]
        # result
        agreement = None
        if not isinstance(self._derivation, DerivationUnavailable):
            agreement = self._decision_frame_copy().get("answer")
        next_step = _next_step_copy(self.run_context, agreement)
        if visual is None:
            return [f"The result cannot be drawn: {self._visual_reason}. {next_step}", "", ""]
        start, end = visual.states[0].mean_px, visual.states[-1].mean_px
        ladder = self._ladder()
        l1 = (f"Corners now land {es.format_px(end)} from where they were found "
              f"({es.format_px(start)} at the start).")
        if ladder is not None and ladder.phase2_rms:
            # SCI3-2: the rise comes from fitting the drawn cube, not from sharing the pose.
            p2, p4 = ladder.phase2_rms, getattr(ladder, "phase4_rms", None) or {}
            below = ("below Phase 2's" if p4 and all(
                cam in p2 and v < p2[cam] for cam, v in p4.items()) else "again")
            l1 += (" One shared pose per photo makes every camera fit the drawn cube, so "
                   "the error rises (B)"
                   + (f"; Phase 4 refines the cube and it falls {below}."
                      if getattr(ladder, "phase4_rms", None) else "."))
        # UX2-5: the next step lives in the Numbers drawer ('Status and next step').
        l2 = self._agreement_sentence()
        del next_step
        l3 = "F: a round, centred cloud means no camera keeps a systematic offset."
        return [l1, l2, l3]

    def _agreement_sentence(self) -> str:
        """L2: what E shows, consistent with its bars (ED-4)."""
        result = self._derivation
        if isinstance(result, DerivationUnavailable) or not getattr(result, "rig_spreads", ()):
            return "Whether the cameras agreed across photos was not measured."
        summary = self._spread_summary()
        answer = self._decision_frame_copy().get("answer")
        if summary is None:
            return f"E: did the cameras agree across photos? {answer}."
        within = summary["within"]
        if answer == "Not clearly" and within and summary["unchecked"]:
            # SCI9 round 9 wording: the bracket closes, and the cameras not yet checked are named
            nums = [str(c + 1) for c in summary["unchecked"]]
            cams = nums[0] if len(nums) == 1 else ", ".join(nums[:-1]) + " and " + nums[-1]
            verb = "has" if len(nums) == 1 else "have"
            camera_word = "camera" if len(nums) == 1 else "cameras"
            their = "its" if len(nums) == 1 else "their"
            head = ("E: turn spreads within 5°" if not summary["judge"]
                    else "E: every spread within its warning level")
            note = "positions not judged; " if not summary["judge"] else ""
            return (f"{head} ({note}{camera_word} {cams} {verb} too few photos, so {their} "
                    "agreement is not yet confirmed).")
        if not summary["judge"]:
            text = ("E: turn spreads within 5° (positions not judged: telecentric)" if within
                    else "E: a turn spread is above 5°")
        else:
            text = ("E: every spread within its warning level" if within else
                    "E: a spread is above its warning level")
        text += f"; overall answer: {answer}."
        return text

    def _spread_summary(self) -> dict | None:
        result = self._derivation
        if isinstance(result, DerivationUnavailable):
            return None
        spreads = getattr(result, "rig_spreads", ()) or ()
        if not spreads:
            return None
        rule = self._position_rule()
        mm = [s.position_scatter_mm for s in spreads if s.position_scatter_mm is not None]
        rot = [float(s.rotation_scatter_deg) for s in spreads]
        mm_known = len(mm) == len(spreads) and rule["judge"]
        within = (max(rot) <= ROTATION_LEVEL_DEG) and (not mm_known or max(mm) <= rule["level_mm"])
        unchecked = [int(s.camera) for s in spreads if s.outlier_rule_status != "applied"]
        return {"max_mm": max(mm) if mm_known else None, "max_deg": max(rot),
                "level_mm": rule["level_mm"], "judge": rule["judge"],
                "within": within, "unchecked": unchecked,
                "max_squares": max((s.position_scatter_squares or 0) for s in spreads)}

    def _position_rule(self) -> dict[str, Any]:
        """How the camera-position spread is judged (SCI2-5 / ED2-12): not at all for a
        telecentric rig; else against 10 mm or 10 % of the target's size, the smaller."""
        visual = self._visual
        if visual is not None and visual.telecentric:
            return {"judge": False, "level_mm": None, "why": _TELECENTRIC_POSITION_WHY}
        level = POSITION_LEVEL_MM
        unit = self._mm_per_unit() if visual is not None else None
        if unit:
            try:
                size_mm = float(_target_geometry(visual)[1]) * float(unit)
                if np.isfinite(size_mm) and size_mm > 0:
                    level = min(level, POSITION_LEVEL_SHARE * size_mm)
            except Exception:
                pass
        return {"judge": True, "level_mm": float(f"{level:.2g}"), "why": ""}

    def _glossary(self, key: str) -> list[str] | None:
        return None

    def _chips(self, key: str) -> list[Chip]:
        visual = self._visual
        chips: list[Chip] = []
        if key == STAGE_GRAPH:
            names = self._camera_names()
            chips.append(Chip(str(len(names)), "cameras"))
            if visual is not None:
                shared = visual.shared_photos()
                n = visual.n_cameras
                pairs = int(np.sum(np.triu(shared, 1) > 0))
                chips.append(Chip(str(len(visual.posed_images)), "photos in the solve"))
                chips.append(Chip(f"{pairs} / {n * (n - 1) // 2}", "camera pairs share photos",
                                  tone="good" if pairs == n * (n - 1) // 2 else None))
            chips.append(Chip(str(len(self._join_events)) if self._join_events else "—",
                              "camera joins" if self._join_events else "join order not recorded"))
            return chips
        if key == STAGE_FIRST:
            result = self._derivation
            start = (visual.states[0].mean_px if visual is not None else
                     getattr(result, "d35_initial_px", None))
            chips.append(Chip(es.format_px(start) if start is not None else "—",
                              "mean error at the start"))
            if visual is not None:
                posed = len(visual.posed_images)
                chips.append(Chip(f"{posed} / {visual.n_images}", "photos placed"))
            excluded = self._excluded_images()
            if excluded and visual is None:  # else 'n / N photos placed' already says it
                chips.append(Chip(str(len(excluded)), "photos left out", tone="warn"))
            check = self._consistency().get("start_matches")
            if check is not None:
                chips.append(Chip("✓" if check else "✗", "matches saved start" if check
                                  else "differs from saved start",
                                  tone="good" if check else "bad"))
            if self._legacy_start():
                chips.append(Chip(es.format_px(result.seed_weighted_mean_px),
                                  "first guess (old run)", tone="warn"))
            return chips
        if key == STAGE_SINGLE:
            recovered, left_out = self._single_face_entries()
            chips.append(Chip(str(len(recovered)), "photos placed", tone="good" if recovered else None))
            chips.append(Chip(str(len(left_out)), "photos left out",
                              tone="warn" if left_out else None))
            fits = [float(e.get("rms_px", np.nan)) for e in recovered]
            fits = [f for f in fits if np.isfinite(f)]
            if fits:
                # UX4-2: one placed photo (or equal ends) shows one value, not "9.56–9.56 px".
                low, high = es.format_px(min(fits)), es.format_px(max(fits))
                value = high if low == high else f"{es.format_number(min(fits))}–{high}"
                chips.append(Chip(value, "fit error range, placed photos"))
            return chips
        if key == STAGE_SOLVER:
            if visual is None:
                return [Chip("not recorded" if "not recorded" in str(
                    self._playback.get("reason", "")) else "unavailable", "solver steps",
                    tone="warn")]
            start, end = visual.states[0].mean_px, visual.states[-1].mean_px
            if self._recorded():
                nits = [s.nit for s in visual.states]
                chips.append(Chip(f"{len(nits) - 1} of {nits[-1]}", "steps recorded"))
            else:
                chips.append(Chip("not recorded", "solver steps", tone="warn"))
            chips.append(Chip(es.format_change(start, end), "mean error", tone="good"
                              if end < start else "warn"))
            if end > 0 and start > 0:
                chips.append(Chip(f"×{es.format_number(start / end)}", "smaller"))
            if self._recorded():  # ED2-14: name the camera that moved most (true size)
                from pyCamSet.workflow.education_visual_p3 import _angle_deg
                turned = _angle_deg(visual.states[0].rotations, visual.states[-1].rotations)
                if turned.size and np.isfinite(turned).any():
                    cam = int(np.nanargmax(turned))
                    chips.append(Chip(f"{es.format_sig(float(turned[cam]))}°",
                                      f"camera {cam + 1} turned most"))
            return chips
        if key == STAGE_LENS:
            changes = self._lens_changes()
            effect = self._lens_effect()
            if effect is not None:
                # ED3-1: lead with the typical move (what B's small gain reflects); the
                # largest move is named with where it is (the picture edge).
                moved = effect.camera_largest_move()
                cam = max(moved, key=moved.get)
                where = self._largest_move_place(effect)
                # The typical move is the chip's own camera's (the one the largest move belongs
                # to), named in the label. Panel A follows the camera picker and names its camera
                # in its title, so the two can differ (SCI5-6).
                typical = float(np.mean(effect.moved_px[effect.rows_camera == cam]))
                chips.append(Chip(f"{es.format_sig(typical)} px",
                                  f"camera {cam + 1}, all photos: typical move · "
                                  f"largest {es.format_sig(moved[cam])} px"
                                  f"{where.replace('picture ', '')}"))
                off = float(np.linalg.norm(effect.off_xy, axis=1).mean())
                on = float(np.linalg.norm(effect.on_xy, axis=1).mean())
                chips.append(Chip(f"{es.format_number(off)} → {es.format_px(on)}",
                                  "error, bending off → on"))
            elif changes:
                shifts = [c.distortion_shift_px for c in changes]
                chips.append(Chip(f"up to {es.format_sig(max(shifts))} px", "distortion shift"))
            later = self._later_lens()
            if later is not None and later.distortion_shift_px:
                chips.append(Chip(f"≤ {es.format_sig(max(later.distortion_shift_px.values()))} px",
                                  "refit in Phase 4"))
            return chips
        # result
        result = self._derivation
        if visual is not None:
            chips.append(Chip(es.format_px(visual.states[-1].mean_px), "mean error",
                              tone="good"))
        summary = self._spread_summary()
        if summary is not None:
            value = (f"{es.format_sig(summary['max_mm'])} mm · "
                     if summary["max_mm"] is not None else "") + \
                f"{es.format_sig(summary['max_deg'])}°"
            label = (f"largest spread (limits {summary['level_mm']:g} mm, 5°)"
                     if summary["max_mm"] is not None else
                     "turn spread (limit 5°)"
                     if not summary["judge"] else "largest turn spread (limit 5°)")
            chips.append(Chip(value, label, tone="good" if summary["within"] else "bad"))
        context = self.run_context
        flags = blocking_reasons(self.metadata)
        status_tone = ("bad" if context.status == "failed" else
                       "good" if (context.status == "complete" and context.quality_check == "checked"
                                  and not flags) else "warn")
        chips.append(Chip(str(context.status), "quality check" if context.quality_check == "checked"
                          else "quality not checked", tone=status_tone))
        check = self._consistency()
        if check.get("final_mean_matches") is False or check.get("final_matches") is False:
            chips.append(Chip("differs", "recomputed vs saved", tone="bad"))
        del result
        return chips

    def _evidence_frame(self, title: str) -> dict[str, Any]:
        return next(frame for frame in self._evidence if frame["title"] == title)

    def _lens_rows(self) -> list[tuple]:
        rows = []
        unit = self._mm_per_unit()
        for c in self._lens_changes():
            name = f"Camera {c.camera + 1}"
            kind = "magnification (px per target unit)" if c.telecentric else "focal length (px)"
            rows.append((name, kind, f"{c.scale_before[0]:.6g}, {c.scale_before[1]:.6g}",
                         f"{c.scale_after[0]:.6g}, {c.scale_after[1]:.6g}"))
            rows.append((name, "image centre (px)",
                         f"{c.centre_before[0]:.1f}, {c.centre_before[1]:.1f}",
                         f"{c.centre_after[0]:.1f}, {c.centre_after[1]:.1f}"))
            rows.append((name, "distortion coefficients",
                         ", ".join(f"{v:.4g}" for v in c.distortion_before) or "none",
                         ", ".join(f"{v:.4g}" for v in c.distortion_after) or "none"))
            rows.append((name, "largest distortion shift in the frame (px)", "0" if not any(
                c.distortion_before) else "—", f"{c.distortion_shift_px:.3g}"))
            if c.telecentric:
                from pyCamSet.workflow.education_visual_p3 import eps_per_metre
                b, a = eps_per_metre(c.eps_before, unit), eps_per_metre(c.eps_after, unit)
                if a is not None and b is not None:
                    rows.append((name, "depth term ε (1/m = % size change per cm)",
                                 f"{b:.3g}", f"{a:.3g}"))
                else:
                    rows.append((name, "depth term ε (per target unit)",
                                 f"{c.eps_before:.4g}", f"{c.eps_after:.4g}"))
                rows.append((name, "bend centre (px)", "",
                             f"{c.centre_after[0]:.1f}, {c.centre_after[1]:.1f}: the principal "
                             "point, the image of the optical axis. A telecentric camera has no "
                             "translation, so the reference photo's target origin lands on this "
                             "point too; its value depends on the frame the run was solved in."))
        effect = self._lens_effect()
        if effect is not None:
            off, on = effect.camera_mean("off"), effect.camera_mean("on")
            moved = effect.camera_largest_move()
            for cam in sorted(on):
                rows.append((f"Camera {cam + 1}", "mean error, bending off / on (px)",
                             f"{off[cam]:.3g}", f"{on[cam]:.3g} (largest corner move "
                             f"{moved[cam]:.3g} px; off = nothing re-fitted)"))
        later = self._later_lens()
        if later is not None:
            for cam in sorted(later.distortion_shift_px):
                eps = later.eps_per_m.get(cam)
                rows.append((f"Camera {cam + 1}", f"Phase 4 run {later.run_id} (same photos, "
                             "target refined)", "",
                             f"distortion shift {later.distortion_shift_px[cam]:.3g} px"
                             + (f"; ε {eps:.3g} /m" if eps is not None else "")))
        return rows

    def _numbers(self, key: str) -> tuple[list[str], list[tuple]]:
        level = self._learning_level
        if key == STAGE_GRAPH:
            chain = self._evidence_frame("Input chain")
            rows = list(chain["rows"])
            edge = self.run_context.verify_input("phase2")
            rows.append(("Lens estimates (Phase 2) input", edge.verification.value, "run record"))
            # SCI3-13: the score in the same unit as panel B (mm when the size is known).
            mm = self._mm_per_unit()
            for event in self._join_events:
                spread = event.get("spread")
                score = ""
                if level >= 2 and spread is not None:
                    score = (f"; disagreement {es.format_sig(float(spread) * mm)} mm"
                             if mm else f"; disagreement {_number(spread)} target units")
                    score += " (position ⊕ target radius × angle)"
                rows.append((f"Camera {int(event.get('joining', 0)) + 1} joined via camera "
                             f"{int(event.get('joined_from', 0)) + 1}",
                             f"{event.get('shared_views', '—')} photos both placed alone",
                             "first-guess assembly, not solver steps" + score))
            if level >= 2:
                rows.append(("About the lens estimates", "", chain["copy"]))
            return ["Item", "Value", "Evidence"], rows
        if key == STAGE_FIRST:
            frame = self._evidence_frame("First guess and solver start")
            rows = list(frame["rows"])
            if level >= 2:
                rows.append(("About the first guess", "", compose(frame, level)))
            return list(frame["headers"]), rows
        if key == STAGE_SINGLE:
            frame = self._evidence_frame(_SINGLE_FACE_TITLE)
            rows = list(frame["rows"])
            if level >= 3:
                rows.append(("Method", "", "", "", _SINGLE_FACE_METHOD))
            return list(frame["headers"]), rows
        if key == STAGE_SOLVER:
            frame = self._evidence_frame("Recorded-step playback")
            rows = list(self._step_rows())
            if level >= 2:
                rows.insert(0, ("About these steps", "", self._playback_copy()))
            if level >= 3 and self._recorded():
                rows.append(("Fixed photo", es.photo_label(self._visual.fixed_pose),
                             "its target pose is fixed in the solve, which removes the rig's "
                             "overall position and turn; the target's known size already fixes "
                             "the scale (the gauge; Phase 4 shows the look-alike changes it "
                             "removes)"))
                rows.append(("Schur picture", "not shown",
                             "the solver also adds Levenberg-Marquardt damping to the camera "
                             "system; the photo poses are then back-substituted"))
            return list(frame["headers"]), rows
        if key == STAGE_LENS:
            return ["Camera", "Number", "Phase 2 (input)", "Phase 3 (result)"], self._lens_rows()
        rows: list[tuple] = []
        ladder = self._ladder()
        if ladder is not None:
            names = self._camera_names()
            for cam in sorted(set(ladder.phase2_rms) | set(ladder.phase3_rms)):
                p4 = ladder.phase4_rms.get(cam) if getattr(ladder, "phase4_rms", None) else None
                rows.append(("Error ladder", names[cam] if cam < len(names) else f"camera {cam + 1}",
                             f"Phase 2 RMS {_number(ladder.phase2_rms.get(cam))} px, "
                             f"Phase 3 RMS {_number(ladder.phase3_rms.get(cam))} px"
                             + (f", Phase 4 RMS {_number(p4)} px" if p4 is not None else ""),
                             ("Phase 2: saved D2.1" if level >= 2 else "Phase 2: saved with that run")
                             + "; Phase 3: recalculated from the solve"))
            if ladder.phase4_mean is not None:
                rows.append(("Error ladder", "Phase 4 mean (not drawn: B shows RMS)",
                             f"{_number(ladder.phase4_mean)} px",
                             f"run {ladder.phase4_run}"
                             + (", saved D4.3" if level >= 2 else ", saved with that run")
                             + ("; its per-camera RMS is replayed from the saved result"
                                if getattr(ladder, "phase4_rms", None) else
                                "; per-camera RMS could not be replayed")))
        for name, count, before, after, vector in self._residual_rows():
            rows.append(("Error per camera / photo", name, f"before {before} px, after {after} px",
                         f"{count} rows; {vector}"))
        decision = self._decision_frame_copy() if not isinstance(
            self._derivation, DerivationUnavailable) else {
                "copy": self._decision_frame_copy()["copy"]}
        if decision.get("answer"):
            rows.append(("Did the cameras agree?", "Overall answer", decision["answer"],
                         "first-guess spreads across photos, relative to camera 1"))
        for item, position, rotation, usable, rule in self._spread_rows():
            rows.append(("Did the cameras agree?", item, f"position {position}, turn {rotation}",
                         f"{usable} usable photos; {rule}"))
        rows.append(("Did the cameras agree?", "Explanation", "", compose(decision, level)))
        agreement = self._agreement() if self._visual is not None else None
        if agreement is not None:
            unit = self._mm_per_unit() or 1.0
            for cam in sorted(agreement.alone):
                rows.append(("Why the cameras must agree", f"camera {cam + 1}, "
                             f"{es.photo_label(agreement.photo)}",
                             f"{_number(agreement.offset[cam] * unit)} "
                             f"{'mm' if self._mm_per_unit() else 'units'}, "
                             f"{_number(agreement.angle_deg[cam])}° from the shared pose",
                             "its own fit with its Phase 2 lens, placed with the first guess's rig"
                             + ("; offset along its own axis not shown (telecentric)"
                                if agreement.depth_removed else "")))
        quality = self._evidence_frame("Quality and next action")
        for setting, value, source in quality["rows"]:
            rows.append(("Quality", setting, value, source))
        rows.append(("Quality", "Status and next step", "", quality["copy"]))
        return ["Topic", "Item", "Value", "Evidence"], rows


def _interp_residual(visual: Any, position: float, rows: np.ndarray) -> np.ndarray:
    from pyCamSet.workflow.education_visual_p3 import photo_points
    observed, predicted, _err = photo_points(visual, position, rows)
    return predicted - observed




def _camera_centres(cameras: np.ndarray) -> np.ndarray | None:
    """Return camera centres C = -R^T t from world-to-camera extrinsics.

    The solver stores each camera as (rotation vector, translation) of the
    world-to-camera transform, so the translation t is not the camera's
    position; a pure rotation changes t while the centre stays put.
    """
    if cameras.ndim == 2 and cameras.shape[1] == 6:
        centres = []
        for camera in cameras:
            rotation, _ = cv2.Rodrigues(np.asarray(camera[:3], dtype=np.float64))
            centres.append(-rotation.T @ camera[3:6])
        return np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    if cameras.ndim == 3 and cameras.shape[1:] == (4, 4):
        return np.linalg.inv(cameras)[:, :3, 3]
    return None


def _largest_camera_step(left: Any, right: Any) -> float | None:
    """Return the largest camera-centre displacement in target coordinate units."""
    try:
        left_cameras = np.asarray(left[1], dtype=float)
        right_cameras = np.asarray(right[1], dtype=float)
        if left_cameras.shape != right_cameras.shape or len(left_cameras) == 0:
            return None
        left_centres = _camera_centres(left_cameras)
        right_centres = _camera_centres(right_cameras)
        if left_centres is None or right_centres is None:
            return None
        delta = right_centres - left_centres
        if not np.all(np.isfinite(delta)):
            return None
        return float(np.max(np.linalg.norm(delta, axis=1)))
    except (IndexError, TypeError, ValueError, np.linalg.LinAlgError, cv2.error):
        return None


def _geometry_snapshot_rows(frame: Any, label: str) -> list[tuple[str, str, str]]:
    """Show camera centres and target-pose translations with state and unit labels."""
    try:
        cameras = np.asarray(frame[1], dtype=np.float64)
        poses = np.asarray(frame[2], dtype=np.float64)
    except (IndexError, TypeError, ValueError):
        return [(f"{label} geometry", "unavailable", "saved frame layout is invalid")]
    if _rotation_only(frame):
        if not np.all(np.isfinite(cameras)):
            return [(f"{label} geometry", "unavailable", "camera orientations are invalid")]
        if not _has_pose_layout(poses) or not np.all(np.isfinite(poses)):
            return [(f"{label} geometry", "unavailable", "target poses are invalid")]
        rows = [(f"{label}: camera {index + 1} orientation", ", ".join(_number(v) for v in camera),
                 "rotation vector, radians; telecentric camera: no projection centre, and its "
                 "translation is not estimated, so no centre is shown")
                for index, camera in enumerate(cameras)]
        rows += [(f"{label}: {_photo_row(index)} target-pose translation",
                  ", ".join(_number(v) for v in _translation(transform)),
                  "target coordinate units; recorded frame")
                 for index, transform in enumerate(poses)]
        return rows
    if not _has_pose_layout(cameras) or not np.all(np.isfinite(cameras)):
        return [(f"{label} geometry", "unavailable", "camera transforms are invalid")]
    if not _has_pose_layout(poses) or not np.all(np.isfinite(poses)):
        return [(f"{label} geometry", "unavailable", "target poses are invalid")]
    rows = []
    try:
        centres = _camera_centres(cameras)
    except (np.linalg.LinAlgError, cv2.error):
        centres = None
    if centres is None or not np.all(np.isfinite(centres)):
        return [(f"{label} geometry", "unavailable", "camera transforms are invalid")]
    for index, centre in enumerate(centres):
        values = ", ".join(_number(value) for value in centre)
        rows.append((f"{label}: camera {index + 1} centre", values,
                     "target coordinate units; C = -Rᵀt from the recorded extrinsics"))
    for index, transform in enumerate(poses):
        values = ", ".join(_number(value) for value in _translation(transform))
        rows.append((f"{label}: {_photo_row(index)} target-pose translation", values,
                     "target coordinate units; recorded frame"))
    return rows


_TELECENTRIC_STEP = "not applicable: telecentric cameras have no centre"


def _rotation_only(frame: Any) -> bool:
    """A telecentric rig's recorded cameras: a rotation vector each, no translation.

    The solver estimates no translation for a telecentric camera (it is not identifiable),
    so its extrinsic block is 3 wide. (On a real telecentric rig the GUI showed every
    Phase 3 geometry row as 'camera transforms are invalid'.)
    """
    try:
        cameras = np.asarray(frame[1], dtype=np.float64)
    except (IndexError, TypeError, ValueError):
        return False
    return cameras.ndim == 2 and cameras.shape[1] == 3 and len(cameras) > 0


def _has_pose_layout(values: np.ndarray) -> bool:
    """Accept the solver's (rotation-vector, translation) or homogeneous layout."""
    return ((values.ndim == 2 and values.shape[1] == 6)
            or (values.ndim == 3 and values.shape[1:] == (4, 4)))


def _translation(value: np.ndarray) -> np.ndarray:
    """Extract translation without converting or comparing rotation parameters."""
    return value[3:6] if value.ndim == 1 else value[:3, 3]


def _photo_row(index: Any) -> str:
    """A photo in the Numbers drawer: the learner's number and the stored 0-based index
    (SCI-8), e.g. "Photo 13 (index 12)"."""
    try:
        value = int(index)
    except (TypeError, ValueError):
        return f"Photo {index}"
    return f"{es.photo_label(value)} (index {value})"


def _one_based(index: Any) -> str:
    """A saved 0-based camera index as its badge number (unknown values pass through)."""
    try:
        return str(int(index) + 1)
    except (TypeError, ValueError):
        return str(index)


def create_education_phase_view(context: Any, parent: QWidget | None = None) -> QWidget:
    """Factory discovered lazily by the shared education dialog."""
    return Phase3View(context, parent)
