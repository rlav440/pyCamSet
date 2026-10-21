"""Teach a saved Phase 4 self-calibration run visually, on the learner's own data.

Design in short: the 3D views use Phase 3's frame and ``fit_view`` call (same viewpoint as
Phase 3's last rig); the solver draws the cube as six face plates (×N) with face trails, lines
back to the design only at L3; the result credits the drop to the shape *together with* the
re-tuned lenses (lens distortion Phase 3 -> 4, "swap one part back", ε in % per cm with sign); the
shape stage reports the alignment-free widths/angles; the trust stage tests the change
against its noise instead of counting "the error went down".

Five stages (stepper pills), each drawn from this run at every detail level:

1. Start from Phase 3 - the rig and target as Phase 4 began, corners coloured by their Phase 3
   error (the shared log error scale), the per-photo error strip plot, and the photos left out.
2. Freeing the shape - the corners moving from the printed design to the refined shape in two
   named parts: first each face as one rigid plate (offset + tilt), then the small change left
   inside each face. The refined shape keeps the printed square size (the scale the exported
   calibration uses) and is only turned + shifted onto the design (SCI-1). A slice through the
   cube shows *why* a hand-assembled cube is never exactly its drawing (ED-7).
3. Watching the solver - the recorded accepted steps replayed in 3D (ScenePlayer, paced by the
   error change) with the error curve in px and per-camera bars; L3 shows which parameter
   block moved at each step.
4. Can we trust it? - the quality gate as traffic-light cards (each says what failing would
   mean), the measurements-vs-unknowns picture (the overfitting risk, ED-8), "moving everything
   together changes no photo" (gauge freedom, ED-6), who saw which photo and the numerical rank
   check (SCI-3: a float64 rank threshold, not a stiffness).
5. Result - Phase 3 vs Phase 4 per-camera error, each camera's lens distortion Phase 3 vs Phase 4
   (the lens model had absorbed the cube's shape error), the "swap one part back" test, every
   photo before/after and (telecentric, L2+) the size change with depth ε (signed, % per cm,
   the unit Phase 3's Lens tuning uses; numerically equal to 1/m).

Data comes from pyCamSet.workflow.education_visual_p4 (Qt-free); the honest statements
(provenance, qualifiers, the decision) live in its Phase4Narrative and appear in the captions,
chips and the Numbers drawer.
"""
from __future__ import annotations

import dataclasses
import math
import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QElapsedTimer, QTimer
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from pyCamSet.gui import education_style as es
from pyCamSet.gui.education_dialog import TransportBar
from pyCamSet.gui.education_worker import EducationWorker
from pyCamSet.gui import education_perf as perf
from pyCamSet.gui.education_scene3d import Scene3D, ScenePlayer, pace_weights
from pyCamSet.gui.education_stage import (
    Chip, EducationStageView, PanelHeading, theme_tokens_for,
)
from pyCamSet.workflow.education_derive import derive_phase4
from pyCamSet.workflow.education_trace import read_trace
from pyCamSet.workflow.education_visual_p4 import (
    EXAGGERATIONS, STAGES, THETA_MIN_DEG, CameraPose, Phase4Narrative, Phase4Visual,
    fmt, plain, plain_blocking, prepare_phase4_visual, step_rows,
)

#: Positions of the printed -> refined morph: 0 printed, 1..10 the faces move as rigid
#: plates, 11..20 the change inside each face is added (adoption: TransportBar(21, ...)).
MORPH_POSITIONS = 21
MORPH_HALF = 10

# --------------------------------------------------------------------------------------
# Display frame: rotate the target frame so the rig's "up" is +z (both 3D backends then
# share one natural view). Display only; every number is computed before this rotation.


def display_rotation(cameras: Sequence[CameraPose]) -> np.ndarray:
    """Rotation Q taking target-frame vectors to a display frame whose +z is the rig's up.

    Up = the mean of the cameras' image-up directions (-row 1 of R); identity without cameras.
    """
    if not cameras:
        return np.eye(3)
    up = -np.mean([np.asarray(c.R)[1] for c in cameras], axis=0)
    norm = float(np.linalg.norm(up))
    if norm < 1e-6:
        return np.eye(3)
    z = up / norm
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - (ref @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def _to_display(Q: np.ndarray, xyz: Any) -> np.ndarray:
    return np.asarray(xyz, dtype=float).reshape(-1, 3) @ Q.T


def _camera_display(Q: np.ndarray, cam: CameraPose) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera (R, t) in the display frame."""
    return np.asarray(cam.R) @ Q.T, np.asarray(cam.t, dtype=float)


def box_edges(points: Any) -> tuple[np.ndarray, np.ndarray]:
    """The 12 edges of the points' axis-aligned box (target frame): (origins, ends)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    pairs = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if bin(a ^ b).count("1") == 1]
    return corners[[a for a, _ in pairs]], corners[[b for _, b in pairs]]


def _nice_clim(values: Any) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return 0.0, 1.0
    hi = float(np.percentile(v, 95))
    return 0.0, hi if hi > 0 else (float(v.max()) or 1.0)


#: SCI7-3 / ED7-2: every measured number on a held run carries this tag (a sentence, chip label
#: or Numbers row that shows a px, µm, mm, % or × value, or a before → after pair).
HELD_TAG = "not verified"  # the tag on numbers; the held lead sentence says "held back"
_MEASURE = re.compile(r"\d[\d,.]*\s*(?:px|µm|mm|%|×|°)|\d[\d.]*\s*(?:→|->)\s*\d")
_BARE_NUMBER = re.compile(r"^\s*[-+]?\d[\d,.]*\s*$")  # a value cell with its unit in the header
#: plain definitions, not results (a pixel's size is a unit, not a measurement); whole words only
_DEFINITION = re.compile(r"(?<![\d.,])(?:1 px|1 square|1 pixel|one pixel)(?![\d.])")
def _lighten_hatches(fig: Any, theme: str) -> None:
    """Round 9 review (contrast): on the Dark theme the hatch of the Phase 3 bars nearly
    vanishes, so its stroke is lightened halfway to white (the fill and the bar are kept)."""
    if str(theme).lower() != "dark":
        return
    from matplotlib.colors import to_rgba
    for ax in fig.axes:
        for patch in ax.patches:
            if patch.get_hatch():
                r, g, b, a = to_rgba(patch.get_edgecolor())
                patch.set_edgecolor((r + (1 - r) * 0.5, g + (1 - g) * 0.5, b + (1 - b) * 0.5, a))


#: SCI9-5 / ED9-3: the one name for the lens quantity (a displacement in px, not a curvature)
LENS_GLOSS = "how far it moves points, in px"


def _measures(text: str) -> bool:
    """True if ``text`` shows a measured number (not a definition) and so needs the tag."""
    return bool(_MEASURE.search(_DEFINITION.sub("", text)))


def _held_text(text: str) -> str:
    """``text`` with the held-back tag placed before its final full stop, once, if it shows a
    measured number (so the tag sits in the same sentence as the number)."""
    if re.search(r"held\s+back|not\s+verified", text, re.IGNORECASE) or not _measures(text):
        return text
    body = text.rstrip()
    if body.endswith("."):
        return f"{body[:-1]} ({HELD_TAG})."
    return f"{body} ({HELD_TAG})"


def _held_caption(text: str) -> str:
    """Each sentence of ``text`` that shows a measured number gets the tag (SCI7-3)."""
    return " ".join(_held_text(part) for part in re.split(r"(?<=[.!?])\s+", text.strip())
                    if part)


def _left_out_label(indices: Sequence[int], limit: int = 8) -> str:
    """``"Photos 10, 13–15 and 18"`` for the photos left out. Counts photos, not list tokens
    (SCI7-2): at most ``limit`` photos are named, and the rest are counted as photos."""
    numbers = sorted({int(i) + 1 for i in indices})  # stored index 0 is "Photo 1"
    if not numbers:
        return "no photos"
    groups: list[list[int]] = [[numbers[0]]]
    for n in numbers[1:]:
        if n == groups[-1][-1] + 1:
            groups[-1].append(n)
        else:
            groups.append([n])
    parts: list[str] = []
    shown = 0
    for group in groups:
        if shown >= limit:
            break
        if len(group) >= 3:
            parts.append(f"{group[0]}–{group[-1]}")
        elif len(group) == 2:
            parts.append(f"{group[0]}, {group[1]}")
        else:
            parts.append(str(group[0]))
        shown += len(group)
    rest = len(numbers) - shown
    if rest:
        parts.append(f"{rest} other photos")  # Round 9: not "more", which reads as a repeat
    word = "Photo" if len(numbers) == 1 else "Photos"
    body = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"{word} {body}"


def _px(value: Any) -> str:
    """A pixel value with two decimals, or "unavailable" (never a fake zero)."""
    text = fmt(value)
    return f"{float(value):.2f}" if text != "unavailable" else text


def _pl(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def _length(value_mm: Any, unit: str) -> str:
    """A length given in mm, written in ``unit`` ("µm" -> whole µm, "mm" -> 2 decimals)."""
    v = float(value_mm)
    if not math.isfinite(v):
        return "unavailable"
    return f"{v * 1000:.0f} µm" if unit == "µm" else f"{v:.2f} mm"


def _signed_length(value_mm: float, unit: str) -> str:
    return ("+" if value_mm >= 0 else "−") + _length(abs(value_mm), unit)


# --------------------------------------------------------------------------------------
# Widgets


class SplitVisual(QWidget):
    """A titled 3D scene (left) beside an optional matplotlib side figure (right).

    The stage view calls :meth:`set_theme`; the Phase 4 view then redraws the stage. The side
    figure uses the education layout engine (aligned, elided panel titles).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("eduP4Split")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)
        left = QWidget(self)
        column = QVBoxLayout(left)
        column.setContentsMargins(4, 0, 0, 0)
        column.setSpacing(0)
        # The shared 3D heading (UX2-19): letter, em-space, bold title, muted subtitle, on the
        # matplotlib panel titles' baseline. ``title``/``subtitle`` read its two lines.
        self.heading = PanelHeading(left)
        self.title = _HeadingLine(self.heading, 0)
        self.subtitle = _HeadingLine(self.heading, 1)
        column.addWidget(self.heading)
        self.scene = Scene3D(left)
        self.scene.setAccessibleName("Phase 4 3D view")
        column.addWidget(self.scene, 1)
        self.figure = Figure(figsize=(5, 4), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        perf.cap_canvas_pixel_ratio(self.canvas)
        self.canvas.setParent(self)
        self.canvas.setMinimumSize(200, 150)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setAccessibleName("Phase 4 side figure")
        outer.addWidget(left, 3)
        outer.addWidget(self.canvas, 2)
        self._left = left
        self._outer = outer
        self._drawer: Callable[[Any, str], None] | None = None
        self._theme = es.active_theme_name()
        self._restyle()

    def set_titles(self, title: str, subtitle: str = "") -> None:
        self.heading.set_text(title, subtitle)
        if isinstance(self.scene, QWidget):  # an OffscreenScene (review renders) is not one
            self.scene.setAccessibleDescription(f"{title}. {subtitle}")

    def set_side(self, drawer: Callable[[Any, str], None] | None,
                 stretch: tuple[int, int] = (3, 2)) -> None:
        """Draw ``drawer(figure, theme)`` in the side figure; None hides it."""
        self._drawer = drawer
        self._outer.setStretch(0, stretch[0])
        self._outer.setStretch(1, stretch[1])
        self.canvas.setVisible(drawer is not None)
        if drawer is not None:
            self.redraw_side()

    def redraw_side(self) -> None:
        import matplotlib
        if self._drawer is None:
            return
        theme = self._theme
        self.figure.clear()
        self.figure._edu_theme = theme
        with matplotlib.rc_context(es.figure_style(theme)):
            self.figure.set_facecolor(es.stage_tokens(theme)["stage"])
            es.align_panel_titles(self.figure)
            self._drawer(self.figure, theme)
        self.canvas.draw_idle()

    def set_theme(self, name: str) -> None:
        self._theme = es.resolve_theme(name)
        self._restyle()
        self.scene.set_theme(self._theme)

    def _restyle(self) -> None:
        self.heading.restyle()


class _HeadingLine:
    """Read access to one line of a :class:`PanelHeading` (0 = "A  Title", 1 = subtitle),
    as the round-1 QLabels offered (``.text()``)."""

    def __init__(self, heading: PanelHeading, line: int) -> None:
        self._heading, self._line = heading, line

    def text(self) -> str:
        return self._heading.heading_text() if self._line == 0 else self._heading._subtitle

    def shown(self) -> str:
        """The line as painted at the current width (elided with "…" when it does not fit)."""
        return self._heading.lines_shown()[self._line]


class MorphBar(TransportBar):
    """The printed -> refined morph of stage 2 on the one transport bar (UX-3).

    21 positions: 0 = printed, 1..10 = each face moving as one rigid plate, 11..20 = the
    change inside each face added. "Exaggerate" (×N, display only) sits behind the bar's ⚙.
    ``on_change(position, factor)`` runs on every move or factor pick. Starts at the end (the
    refined shape); never plays on its own.
    """

    def __init__(self, on_change: Callable[[int, int], None], factors: Sequence[int],
                 default: int, describe: Callable[[int], str], short: Callable[[int], str],
                 parent: QWidget | None = None) -> None:
        super().__init__(MORPH_POSITIONS, parent, describe=describe, short=short,
                         unit_label="Printed → refined", interval_ms=60, min_interval_ms=40,
                         speeds=None, start=MORPH_POSITIONS - 1,
                         scrub_help="The start is the printed design, the end the refined shape "
                                    "(corrections exaggerated, display only).",
                         play_name="Play printed → refined")
        self._on_change = on_change
        self._factors = [int(f) for f in factors]
        self._factor_index = self._factors.index(default) if default in self._factors else 0
        self.exaggerate_menu = self.add_menu_choice(
            "Exaggerate", [f"×{f}" for f in self._factors], self._factor_index,
            self._factor_picked)
        self.positionChanged.connect(lambda _p: self._on_change(self.position(),
                                                                self.exaggeration))

    @property
    def exaggeration(self) -> int:
        return int(self._factors[self._factor_index])

    @property
    def factors(self) -> list[int]:
        return list(self._factors)

    @property
    def fraction(self) -> float:
        return self.position() / (MORPH_POSITIONS - 1)

    def set_exaggeration(self, factor: int) -> None:
        """Pick ``factor`` (one of :attr:`factors`), as the ⚙ menu does."""
        self._factor_picked(self._factors.index(int(factor)))
        for k, action in enumerate(self.exaggerate_menu.actions()):
            action.setChecked(k == self._factor_index)

    def _factor_picked(self, index: int) -> None:
        if index == self._factor_index:
            return
        self._factor_index = int(index)
        self._on_change(self.position(), self.exaggeration)

    def set_theme(self, _name: str) -> None:  # the stage view tells controls about themes
        pass


# --------------------------------------------------------------------------------------
# Figure helpers (matplotlib; used by the view and by the review renderer)


def visible_camera_colour(cam: int, theme: str) -> str:
    """Camera colour for the coverage raster's shading: the shared per-theme camera fill
    (``education_style.camera_fill``, UX2-1; moved to the shared layer in round 2)."""
    return es.camera_fill(cam, theme)


def _tone_colour(t: Mapping[str, str], tone: str) -> str:
    return {"good": t["good"], "warn": t["warn"], "bad": t["bad"]}.get(tone, t["muted"])


_TONE_MARK = {"good": "✓", "warn": "!", "bad": "✕"}


def draw_gate_cards(fig: Any, spec: Any, checks: Sequence[Any], theme: str, *,
                    compact: bool = False) -> None:
    """A row of traffic-light cards (one per check) in gridspec slot ``spec``: title, value,
    what was found, and (muted) what failing would mean (ED-8). Every finding is written with
    ``es.card_text``, so it wraps and shrinks to stay inside its card (UX3-1)."""
    from matplotlib.patches import FancyBboxPatch
    t = es.stage_tokens(theme)
    tok = theme_tokens_for(theme)
    sub = spec.subgridspec(1, max(len(checks), 1), wspace=0.04)
    for k, check in enumerate(checks):
        ax = fig.add_subplot(sub[0, k])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        colour = _tone_colour(t, check.tone)
        card = FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                              boxstyle="round,pad=0,rounding_size=0.05",
                              facecolor=tok["surface_alt"], edgecolor=tok["border"],
                              linewidth=1.0, transform=ax.transAxes, clip_on=False)
        ax.add_patch(card)
        ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.03, 0.96,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    facecolor=colour, edgecolor="none",
                                    transform=ax.transAxes, clip_on=False))
        mark_y = 0.80
        ax.text(0.10, mark_y, _TONE_MARK.get(check.tone, "?"), ha="center", va="center",
                fontsize=8, fontweight="bold",
                color=tok["on_accent"] if check.tone in _TONE_MARK else t["stage"],
                transform=ax.transAxes,
                bbox=dict(boxstyle="circle,pad=0.15", facecolor=colour, edgecolor="none"))
        if compact:  # L3: title, value, then the finding wrapped inside the card
            ax.text(0.18, mark_y, check.title, ha="left", va="center", fontsize=9.5,
                    fontweight="bold", color=t["text"], transform=ax.transAxes)
            ax.text(0.08, 0.56, check.value, ha="left", va="center", fontsize=9.2,
                    fontweight="bold", color=colour if check.tone in _TONE_MARK else t["text"],
                    transform=ax.transAxes)
            es.card_text(ax, 0.08, 0.38, check.line, card=card, transform=ax.transAxes,
                         ha="left", va="top", fontsize=8.4, color=t["text"], max_lines=2)
            continue
        ax.text(0.18, mark_y, check.title, ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=t["text"], transform=ax.transAxes)
        # Round 9: a held value carries "(not verified)", so it is set smaller to fit its card
        # UX-layout (round 10 item 4): the value is fitted to its card, so "Converged (not
        # judged)" shrinks inside the card instead of running past its edge.
        es.card_text(ax, 0.08, 0.59, check.value, card=card, max_lines=1, ha="left", va="center",
                     fontsize=11.5 if "(not verified)" in check.value else 14,
                     fontweight="bold", color=colour if check.tone in _TONE_MARK else t["text"],
                     transform=ax.transAxes)
        # Round 10 item 4: the body starts a little higher and is a touch smaller, so its second
        # line clears the italic guard line at the card's foot (1280 px cards are short)
        es.card_text(ax, 0.08, 0.44, check.line, card=card, transform=ax.transAxes,
                     ha="left", va="top", fontsize=8.2, color=t["text"], max_lines=2)
        guard = getattr(check, "guard", "")
        if guard:
            # Round 9 review: the guard line in the body colour at 11 pt (was muted, 8.2 pt).
            es.card_text(ax, 0.08, 0.10, guard, card=card, transform=ax.transAxes,
                         ha="left", va="center", fontsize=10, color=t["text"],
                         style="italic", max_lines=1)


def draw_coverage(ax: Any, coverage: np.ndarray, excluded: Sequence[Any], theme: str, *,
                  badge_size: float = 9.0) -> None:
    """Who saw which photo: one row per camera (its colour, darker = more corners), one
    column per photo (numbered from 1); left-out photos outlined in red with their number in
    a band on top of the raster (UX-22), never on the tick row."""
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Rectangle
    t = es.stage_tokens(theme)
    n_images, n_cams = coverage.shape
    peak = max(int(coverage.max()), 1)
    stage = np.array(to_rgb(t["stage"]))
    img = np.zeros((n_cams, n_images, 3))
    for c in range(n_cams):
        colour = np.array(to_rgb(visible_camera_colour(c, theme)))
        share = np.sqrt(np.clip(coverage[:, c] / peak, 0, 1))
        alpha = np.where(coverage[:, c] > 0, 0.25 + 0.75 * share, 0.0)
        img[c] = stage[None, :] * (1 - alpha[:, None]) + colour[None, :] * alpha[:, None]
    ax.imshow(img, aspect="auto", interpolation="nearest",
              extent=(-0.5, n_images - 0.5, n_cams - 0.5, -0.5))
    band = 1.3 if excluded else 0.0
    for ex in excluded:
        ax.add_patch(Rectangle((ex.image - 0.5, -0.5), 1.0, n_cams, fill=False,
                               edgecolor=t["bad"], linewidth=1.8, zorder=4))
    # Neighbouring left-out photos share one label ("13–15"), so numbers never collide.
    near = max(2, n_images // 25)
    groups: list[list[int]] = []
    for image in sorted(ex.image for ex in excluded):
        if groups and image - groups[-1][-1] <= near:
            groups[-1].append(image)
        else:
            groups.append([image])
    for group in groups:
        first, last = es.photo_number(group[0]), es.photo_number(group[-1])
        text = str(first) if len(group) == 1 else (
            f"{first}, {last}" if len(group) == 2 else f"{first}–{last}")
        ax.text((group[0] + group[-1]) / 2, -0.5 - band / 2, text, ha="center",
                va="center", fontsize=8.5, color=t["bad"], fontweight="bold", zorder=5)
    ax.set_ylim(n_cams - 0.5, -0.5 - band)
    ax.set_yticks(range(n_cams))
    ax.set_yticklabels([es.camera_badge_label(c) for c in range(n_cams)], fontsize=badge_size,
                       fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=7)
    for label, cam in zip(ax.get_yticklabels(), range(n_cams)):
        label.set_color(es.badge_text_colour(cam, ax))
        label.set_bbox(dict(boxstyle=f"circle,pad={0.25 * badge_size / 9:.3f}",
                            **es.camera_mark_kwargs(cam, ax)))
    step = 10 if n_images > 30 else 5 if n_images > 10 else 1
    numbers = [1] + list(range(step, n_images + 1, step)) if step > 1 else \
        list(range(1, n_images + 1))
    ax.set_xticks([n - 1 for n in numbers], [str(n) for n in numbers])
    ax.set_xlabel("Photo")
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_excluded_cards(ax: Any, excluded: Sequence[Any], theme: str, *, detail: bool) -> None:
    """The photos left out: a red-ringed photo number (from 1), the camera badge(s) that saw it."""
    t = es.stage_tokens(theme)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if not excluded:
        ax.text(0.0, 0.5, "✓  every photo had a target position", color=t["good"],
                fontsize=10.5, va="center", transform=ax.transAxes)
        return
    shown = list(excluded)[:6]
    cols = min(len(shown), 2 if detail else 3)
    rows = int(math.ceil(len(shown) / cols))
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
    # Round 9 review: the "+N more" count gets a line of its own under the rows, so it cannot
    # run into the last row's text.
    more = len(excluded) - len(shown)
    floor = 0.26 if more > 0 else 0.0

    def area(text: str, **kw: Any) -> TextArea:
        return TextArea(text, textprops=dict(fontsize=kw.pop("fontsize", 9), **kw))

    for k, ex in enumerate(shown):
        r, c = divmod(k, cols)
        y = 1 - (r + 0.5) * (1 - floor) / rows  # rows fill the space above the floor
        parts = [area(es.photo_label(ex.image, short=True), fontweight="bold", color=t["bad"],
                      fontsize=9.5, bbox=dict(boxstyle="circle,pad=0.35", facecolor=t["stage"],
                                              edgecolor=t["bad"], linewidth=2.0)),
                 area(" seen by" if ex.cameras else " no corners found", color=t["muted"])]
        for cam in ex.cameras[:3]:
            parts.append(area(es.camera_badge_label(cam), fontweight="bold", fontsize=8.5,
                              color=es.badge_text_colour(cam, ax),
                              bbox=dict(boxstyle="circle,pad=0.25",
                                        **es.camera_mark_kwargs(cam, ax))))
        tail = ("only" if len(ex.cameras) == 1 else "") + (
            f" · {ex.total} corners" if detail and ex.cameras else "")
        if tail:
            parts.append(area(tail.strip(" ·") if not tail.startswith("only") else tail,
                              color=t["muted"]))
        box = HPacker(children=parts, align="center", pad=0, sep=7)
        ax.add_artist(AnnotationBbox(box, (c / cols + 0.01, y), xycoords="axes fraction",
                                     box_alignment=(0.0, 0.5), frameon=False, pad=0))
    if more > 0:
        ax.text(0.0, 0.0, f"+{more} more", ha="left", va="bottom",
                fontsize=8.8, color=t["muted"], transform=ax.transAxes)


def _plain_log_ticks(ax: Any, lo: float, hi: float, axes: str = "xy") -> None:
    """Log axes with plain tick labels (0.2, 0.5, 1, 2, 5 ...) instead of 2 x 10^-1."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ticks = [m * 10.0 ** e for e in range(-4, 5) for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
    fmt_tick = FuncFormatter(lambda v, _p: f"{v:g}")
    for axis in [a for name, a in (("x", ax.xaxis), ("y", ax.yaxis)) if name in axes]:
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(fmt_tick)
        axis.set_minor_locator(NullLocator())


def draw_nullity_equation(ax: Any, nullity: Mapping[str, Any], theme: str) -> None:
    """``unknowns = determined by the photos + corners with one detection + weakly fixed lens or
    pose directions`` as numbers (L3). The symmetries are already removed from the unknowns, so
    they are not here. The last term counts null directions of the lens and pose unknowns after
    the corners are eliminated; it is not a motion of the rig."""
    t = es.stage_tokens(theme)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    total = nullity.get("free_parameters")
    terms = [(total, "unknowns", t["text"]),
             (nullity.get("rank"), "determined by\nthe photos", t["good"]),
             (nullity.get("point_local_nullity"), "corners with\none detection", t["warn"]),
             (nullity.get("global_nullity"), "weak lens or\npose directions", t["bad"])]
    xs = [0.11, 0.40, 0.66, 0.90]
    for k, ((value, label, colour), x) in enumerate(zip(terms, xs)):
        ax.text(x, 0.66, "?" if value is None else f"{int(value):,}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=colour, transform=ax.transAxes)
        ax.text(x, 0.16, label, ha="center", va="center", fontsize=8.5, color=t["muted"],
                transform=ax.transAxes, linespacing=1.15)
        if k < 3:
            ax.text((x + xs[k + 1]) / 2 + (0.01 if k else 0.0), 0.66, "=" if k == 0 else "+",
                    ha="center", va="center", fontsize=13, color=t["muted"],
                    transform=ax.transAxes)


def draw_unknowns(ax: Any, visual: Phase4Visual, theme: str, *, compact: bool = False,
                  equation: bool = False, define: bool = False, short: bool = False) -> bool:
    """Measured numbers vs unknowns as one proportional bar (ED-8 / UX2-16): the measured
    numbers as a grey bar, the unknowns as the accent sliver at its start, split by kind and
    named in one line. False when not recorded."""
    t = es.stage_tokens(theme)
    nullity = visual.nullity or {}
    shape = nullity.get("jacobian_shape") or ()
    measured = int(shape[0]) if len(shape) == 2 else None
    unknowns = nullity.get("free_parameters") or (int(shape[1]) if len(shape) == 2 else None)
    if not measured or not unknowns:
        es.empty_panel(ax, "The number of measurements and unknowns was not recorded.")
        return False
    unknowns = int(unknowns)
    ax.barh([0], [measured], height=0.8, color=t["muted"], alpha=0.45, zorder=3)
    sizes = visual.block_sizes
    tele = bool(visual.telecentric) and all(visual.telecentric)
    names = ("lenses", "camera turns" if tele else "camera poses", "target in each photo",
             "corners")
    shades = (1.0, 0.72, 0.48, 0.28)
    parts = []
    if sizes is not None and sum(sizes) == unknowns:
        left = 0
        for size, name, shade in zip(sizes, names, shades):
            ax.barh([0], [size], left=left, height=0.8, color=t["accent"], alpha=shade,
                    zorder=4)
            parts.append(f"{name} {size:,}")
            left += size
    else:
        ax.barh([0], [unknowns], height=0.8, color=t["accent"], zorder=4)
    # Round 9 review: the L2 panel is narrower, so its label drops the (x, y) gloss (kept at L1, L3)
    # In the narrow (short) panel the label sits just past the bar's end, so it stays in the panel.
    ax.annotate(f"{measured:,} measured" if compact else
                f"{measured:,} measured numbers" if (define or short) else
                f"{measured:,} measured numbers (x, y of every detected corner)",
                (measured, 0), xytext=(6, 0) if short else (-6, 0), textcoords="offset points",
                ha="left" if short else "right",
                va="center", fontsize=9, color=t["text"], zorder=5)
    # ED6-2: the word is defined on the line that counts it where that line has the width
    # (full-width panels only; a half-width L2 panel would run into its neighbour)
    gloss = " (the numbers the solver can change)" if define or equation else ""
    # Round 9: the same "not pinned down" words as the stage 4 caption, wherever the count shows.
    rank_value = nullity.get("rank")
    gap = unknowns - int(rank_value) if rank_value is not None else 0
    not_pinned = (f", of which {gap:,} are not pinned down (rank {int(rank_value):,})"
                  if gap > 0 and not compact else "")
    if short and parts and not compact:
        # Round 9 review: the narrow panel puts the parts on a second line (no run into panel E)
        label = f"{unknowns:,} unknowns{gloss}{not_pinned}:\n" + " · ".join(parts)
    else:
        label = (f"{unknowns:,} unknowns{gloss}{not_pinned}"
                 + (": " + " · ".join(parts) if parts and not compact else ""))
    if equation and nullity.get("rank") is not None:
        # L3: where the unknowns go (the rank check's bookkeeping) in one line. Round 9: the
        # unknowns not pinned down are named with the same words as the stage 4 caption.
        rank = int(nullity["rank"])
        gap = unknowns - rank
        not_pinned = (f", of which {gap:,} are not pinned down (rank {rank:,})" if gap > 0 else "")
        label = (f"{unknowns:,} unknowns{gloss}{not_pinned}: "
                 f"{rank:,} determined by the photos + "
                 f"{int(nullity.get('point_local_nullity') or 0)} corners with one detection "
                 "(may slide) + "
                 f"{int(nullity.get('global_nullity') or 0)} weakly fixed lens or pose directions")
    ax.annotate(label,
                (0, -0.4), xytext=(0, -4), textcoords="offset points", ha="left", va="top",
                fontsize=8.8, color=t["accent"], fontweight="bold", annotation_clip=False)
    ax.set_xlim(0, measured * 1.005)
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    return True


def noise_rows(visual: Phase4Visual, unit: str) -> list[tuple[str, float, float, str]]:
    """Rows of the "bigger than its noise?" picture (ED-6 / SCI2-8): (label, low, high, kind)
    in ``unit``; kind "noise" or "change". [] without a formal precision."""
    _sd3, noise_mean = visual.noise_3d()
    if noise_mean is None:
        return []
    k = 1000.0 if unit == "µm" else 1.0
    rows = [("noise alone (expected)", noise_mean * k, noise_mean * k, "noise")]
    summary = visual.face_summary()
    if summary is not None:
        rows.append(("change inside the faces", summary["inside_mean"] * k,
                     summary["inside_mean"] * k, "change"))
        pairs = visual.pairs()
        if pairs:
            rows.append(("cube wider than drawn", min(p.change for p in pairs) * k,
                         max(p.change for p in pairs) * k, "change"))
    else:
        sizes = visual.correction_mm()
        several = list(visual.several)
        if sizes is not None and several:
            mean = float(np.nanmean(sizes[several])) * k
            rows.append(("mean corner change", mean, mean, "change"))
    return rows


def draw_noise_check(ax: Any, visual: Phase4Visual, theme: str, unit: str,
                     note: str = "") -> bool:
    """ED2-6: is the shape change bigger than its noise? Each change (inside the faces, the
    face moves, the opposite-face totals) against the 3D distance noise alone would give,
    from the corners' formal precision. Log axis; ranges drawn as a bar from low to high.
    ``note`` (ED9-1) is a plain line under the axis that says what the cube bar means."""
    t = es.stage_tokens(theme)
    rows = noise_rows(visual, unit)
    if len(rows) < 2:
        es.empty_panel(ax, "The corners' precision could not be computed for this run, so the "
                           "change is not compared with its noise.")
        return False
    noise = rows[0][1]
    ys = np.arange(len(rows))
    lo_all = min(r[1] for r in rows)
    for y, (label, lo, hi, kind) in zip(ys, rows):
        colour = t["muted"] if kind == "noise" else t["accent"]
        start = lo_all / 3
        ax.barh([y], [hi - start], left=start, height=0.5, color=colour,
                alpha=0.35 if lo != hi else 0.85, zorder=3)
        if lo != hi:
            ax.barh([y], [hi - lo], left=lo, height=0.5, color=colour, alpha=0.9, zorder=4)
        value = (f"{lo:.0f}–{hi:.0f}" if lo != hi else f"{hi:.0f}") if unit == "µm" else \
            (f"{lo:.2f}–{hi:.2f}" if lo != hi else f"{hi:.2f}")
        # ED5-8: the caption's rounding (down to one decimal, never up: 4.7x, not 5x)
        mult = lo if lo != hi else hi
        ratio = "" if kind == "noise" else f"  ({math.floor(mult / noise * 10) / 10:.1f}× noise)"
        # Round 9 review: the cube bar says what it measures, in the figure (the caption is at its limit)
        wider = " wider than drawn" if label == "cube wider than drawn" else ""
        ax.annotate(f"{value} {unit}{wider}{ratio}", (hi, y), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.8, color=t["text"], zorder=5)
    ax.axvline(noise, color=t["muted"], linestyle=(0, (3, 2)), linewidth=1.2, zorder=2)
    ax.set_xscale("log")
    hi_all = max(r[2] for r in rows)
    ax.set_xlim(lo_all / 3, hi_all * 12)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([m * 10.0 ** e for e in range(-3, 6) for m in (1, 3)
                                             if lo_all / 3 <= m * 10.0 ** e <= hi_all * 12]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _p: f"{val:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if note:
        ax.set_xlabel(note, fontsize=8.5, color=t["muted"], linespacing=1.3)
    return True


_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def held_corners_copy(visual: Phase4Visual) -> tuple[str | None, int | None]:
    """How many corners the solver fixed, read from the run's own gauge indices (D4.2).

    Returns the count in words for the copy and the number of coordinates those corners fix:
    two corners in full and one coordinate of the third, 3 + 3 + 1 = 7. The coordinate count
    is None unless the run recorded exactly three corners. With no recorded indices the count
    is None too, and the copy says "the corners" without a number: no count is assumed."""
    n = len(visual.gauge_indices or ())
    if not n:
        return None, None
    return _COUNT_WORDS.get(n, str(n)), (3 * (n - 1) + 1 if n == 3 else None)


def look_alike_subtitle(tele: bool, count: int | None) -> str:
    """Subtitle of panel C (L1 and up): which changes look identical, that no photo changes,
    and what the solver fixed. ``count`` is the number of corners the run recorded as fixed,
    or None when it recorded none (then no number is written). Kept short: it must fit one
    line at 1280 px."""
    moves = "turned or scaled" if tele else "moved, turned or scaled"
    # the pinhole line is longer, so it says "one pose" (the telecentric one names the photo)
    pose = "one photo's pose" if tele else "one pose"
    fixed = f"{count} corners fixed" if count else "the corners fixed"
    return f"{moves}: no photo changes; {pose} and {fixed}"


def draw_gauge_strip(fig: Any, spec: Any, visual: Phase4Visual, Q: np.ndarray, theme: str, *,
                     letter: str = "C", compact: bool = False) -> None:
    """Gauge freedom (symmetries) as three small top views of this rig (ED-6): as solved,
    everything turned together, everything made bigger. Each makes exactly the same photos.
    The solver removes these look-alikes by fixing one photo's pose and the printed positions of
    the run's reference corners, so it solves for one specific form.

    SCI2-7 / SCI2-2: for telecentric cameras the size variant keeps the cameras where they
    are and divides the lens magnification and telecentricity by 1.3 (a camera's distance is
    not seen); the turn is exact, a sideways shift is only nearly exact for this lens model, so
    it is not drawn as a free move; no error value is printed under the variants (none was
    recomputed)."""
    t = es.stage_tokens(theme)
    state = visual.final or visual.start
    sub = spec.subgridspec(1, 3, wspace=0.06)
    cams = state.cameras if state is not None else ()
    centre = np.zeros(2)
    dirs = []
    for cam in cams:
        d = (Q @ (cam.centre - (np.nanmean(visual.printed, axis=0)
                                if visual.printed is not None else 0)))[:2]
        n = float(np.linalg.norm(d))
        dirs.append(d / n if n > 0 else np.array([1.0, 0.0]))
    tele = bool(visual.telecentric) and all(visual.telecentric)
    variants = (("as solved", "", 0.0, 1.0, 1.0),
                ("all turned", "", 35.0, 1.0, 1.0),
                ("all 30 % bigger", "lenses 1.3× weaker" if tele else "cameras 30 % further",
                 0.0, 1.3, 1.0 if tele else 1.3))
    for k, (label, note, turn, grow, cam_grow) in enumerate(variants):
        ax = fig.add_subplot(sub[0, k])
        ax.set_aspect("equal")
        ax.set_anchor("N")  # top-aligned, so the panel title lines up with its row
        ax.set_axis_off()
        a = math.radians(turn)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        square = 0.62 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
        sq = (square * grow) @ rot.T
        ax.fill(sq[:, 0], sq[:, 1], color=t["muted"], alpha=0.25, zorder=2)
        ax.plot(sq[:, 0], sq[:, 1], color=t["text"], linewidth=1.2, zorder=3)
        for cam, d in zip(cams, dirs):
            p = (d * 1.75 * cam_grow) @ rot.T
            ax.plot([p[0], centre[0]], [p[1], centre[1]],
                    **{**es.camera_mark_kwargs(cam.index, ax, "line"), "linewidth": 1.0},
                    alpha=0.8, zorder=2)
            # coloured dots: badges would collide at this size; the colours are the key
            ax.scatter([p[0]], [p[1]], s=42, zorder=4,
                       **es.camera_mark_kwargs(cam.index, ax, "dot"))
        ax.set_xlim(-2.6, 2.95)
        ax.set_ylim(-2.6, 2.6)
        ax.annotate(label, (0.5, 0.0), xycoords="axes fraction", xytext=(0, -1),
                    textcoords="offset points", ha="center", va="top", fontsize=9,
                    fontweight="bold", color=t["text"])
        if note or not compact:
            ax.annotate(note or "same photos", (0.5, 0.0), xycoords="axes fraction",
                        xytext=(0, -15), textcoords="offset points", ha="center", va="top",
                        fontsize=8.5, color=t["good"] if not note else t["text"])
        if k == 0:
            # ED4-10: say what the look-alike changes are, and how the solver removes them: it
            # fixes one photo's pose and the run's own reference corners, so one form is solved.
            # The subtitle must fit 1280 px in one line. A sideways shift is only nearly exact
            # for telecentric lenses, so it is not claimed there.
            subtitle = look_alike_subtitle(tele, len(visual.gauge_indices or ()))
            es.panel_title(ax, letter, "Some changes look identical in every photo", subtitle)


def _rms_of(table: Any) -> float | None:
    """Root-mean-square of one error table's per-observation errors (px), or None."""
    norms = getattr(table, "norms", None)
    if norms is None or not len(norms):
        return None
    return float(np.sqrt(np.mean(np.square(np.asarray(norms, dtype=float)))))


def draw_rank_check(ax: Any, nullity: Mapping[str, Any], theme: str, letter: str) -> bool:
    """The smallest singular values of the reduced Jacobian against the float64 rank
    threshold (SCI-3): a numerical-rank check in mixed units, not a stiffness."""
    t = es.stage_tokens(theme)
    values = list(nullity.get("smallest_singular_values") or [])
    tol = nullity.get("tolerance")
    if not values or tol is None:
        es.empty_panel(ax, "The rank check's values were not recorded.")
        es.panel_title(ax, letter, "Numerical rank check", None)
        return False
    vals = np.sort(np.asarray(values, dtype=float))
    ax.bar(range(len(vals)), vals, color=[t["good"] if x > float(tol) else t["bad"]
                                          for x in vals], width=0.65, zorder=3)
    es.limit_line(ax, float(tol), values=vals, above_is_bad=False)
    ax.set_yscale("log")
    ax.set_ylim(float(tol) / 4, max(vals.max(), float(tol)) * 3)
    ax.set_xticks(range(len(vals)), [str(i + 1) for i in range(len(vals))])
    ax.set_xlabel("weakest directions, smallest first")
    ax.set_ylabel("singular value (mixed units)")
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    es.panel_title(ax, letter, "Numerical rank check",
                   "dashes: float64 rank threshold (free: below)")
    return True


def draw_face_slice(ax: Any, visual: Phase4Visual, Q: np.ndarray, factor: int, theme: str,
                    unit: str, tag: str = "") -> bool:
    """A slice through the cube (ED-7), looking down the cube axis closest to the rig's up:
    the faces around it as designed (dashed) and as solved (solid plate: the face's rigid
    move, ×``factor``), each with this run's outward offset and tilt. False when the target
    has no ring of faces to show."""
    t = es.stage_tokens(theme)
    if not visual.faces or visual.plate is None or visual.printed is None:
        return False
    up = Q.T @ np.array([0.0, 0.0, 1.0])
    # ED2-23: slice through the face that moved most (then the one nearest the rig's up).
    most = max(visual.faces, key=lambda f: abs(f.offset))
    candidates = [f.normal for f in visual.faces if abs(float(f.normal @ most.normal)) < 0.5]
    axis = max(candidates or [f.normal for f in visual.faces],
               key=lambda n: abs(float(n @ up)))
    axis = axis * (np.sign(axis @ up) or 1.0)
    side = [f for f in visual.faces if abs(float(f.normal @ axis)) < 0.5]
    if len(side) < 2:
        return False
    e1 = side[0].normal - (side[0].normal @ axis) * axis
    e1 = e1 / (np.linalg.norm(e1) or 1.0)
    basis = np.vstack([e1, np.cross(axis, e1)])  # looking down the axis
    printed, plate = visual.printed, visual.plate

    def flat(xyz: np.ndarray) -> np.ndarray:
        return np.asarray(xyz, dtype=float) @ basis.T

    centres, design_ends = [], []
    size = float(np.linalg.norm(np.ptp(flat(printed[np.concatenate(
        [np.asarray(f.points) for f in side])]), axis=0)))
    for face in side:
        idx = np.asarray(face.points, dtype=int)
        n2 = flat(face.normal)
        n2 = n2 / (np.linalg.norm(n2) or 1.0)
        along = np.array([-n2[1], n2[0]])
        d = flat(printed[idx])
        m = flat(printed[idx] + factor * (plate[idx] - printed[idx]))
        cd, cm = d.mean(axis=0), m.mean(axis=0)
        half = (d @ along).max() - (d @ along).min()
        ends_d = np.array([cd - along * half / 2, cd + along * half / 2])
        _u, _s, vt = np.linalg.svd(m - cm)
        dir_m = vt[0] * (np.sign(vt[0] @ along) or 1.0)
        span = (m - cm) @ dir_m
        ends_m = np.array([cm + dir_m * span.min(), cm + dir_m * span.max()])
        ax.plot(ends_d[:, 0], ends_d[:, 1], color=t["muted"], linestyle=(0, (4, 3)),
                linewidth=1.3, zorder=3)
        ax.plot(ends_m[:, 0], ends_m[:, 1], color=t["text"], linewidth=4.0,
                solid_capstyle="butt", zorder=4)
        ax.annotate("", xy=cm, xytext=cd, arrowprops=dict(arrowstyle="-|>", color=t["accent"],
                                                          lw=1.4, shrinkA=0, shrinkB=0),
                    zorder=5)
        label_at = cm + n2 * 0.12 * size
        es.draw_face_tags(ax, [label_at], [face.face], size=8.5)
        ha = "left" if n2[0] > 0.45 else "right" if n2[0] < -0.45 else "center"
        va = "bottom" if n2[1] > 0.45 else "top" if n2[1] < -0.45 else "center"
        side_px = (14 * np.sign(n2[0]) if ha != "center" else 0,
                   12 * np.sign(n2[1]) if va != "center" else 0)
        ax.annotate(f"{_signed_length(face.offset, unit)} · {face.tilt_deg:.1f}°{tag}", label_at,
                    xytext=side_px, textcoords="offset points", ha=ha, va=va, fontsize=8.5,
                    color=t["text"], zorder=6)
        centres.append(cd)
        design_ends.append(ends_d)
    pts = np.vstack(design_ends)
    middle = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - middle[1], pts[:, 0] - middle[0]))
    block = pts[order]
    ax.fill(block[:, 0], block[:, 1], color=t["muted"], alpha=0.12, zorder=1)
    ax.text(middle[0], middle[1], "design", ha="center", va="center",
            fontsize=8.5, color=t["muted"], zorder=2, multialignment="center")
    # UX4-4: a wider frame lets the slice fill the panel's height (labels sit within 0.8).
    ax.set_xlim(middle[0] - 1.0 * size, middle[0] + 1.0 * size)
    ax.set_ylim(middle[1] - 0.90 * size, middle[1] + 0.90 * size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return True


def draw_face_bars(ax: Any, faces: Sequence[Any], theme: str, unit: str, tag: str = "") -> None:
    """Each face as one rigid plate: its outward move (bar) and tilt (label)."""
    t = es.stage_tokens(theme)
    factor = 1000.0 if unit == "µm" else 1.0
    vals = np.array([f.offset * factor for f in faces])
    ys = np.arange(len(faces))
    ax.barh(ys, vals, height=0.62, color=t["accent"], zorder=3)
    ax.axvline(0, color=t["muted"], linewidth=1.0, zorder=2)
    for y, f, v in zip(ys, faces, vals):
        ax.annotate((f"{v:+.0f} · {f.tilt_deg:.1f}°" if unit == "µm" else f"{v:+.2f} · "
                     f"{f.tilt_deg:.1f}°") + tag, (v, y), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", ha="left" if v >= 0 else "right",
                    va="center", fontsize=8.8, color=t["text"])
    es.face_ticks(ax, ys, [f.face for f in faces], axis="y", size=8.5)
    ax.invert_yaxis()
    lo, hi = min(0.0, float(vals.min())), max(0.0, float(vals.max()))
    span = (hi - lo) or 1.0
    ax.set_xlim(lo - (0.45 * span if lo < 0 else 0.02 * span), hi + 0.55 * span)
    ax.set_xlabel(f"face moved outward ({unit}) · tilt (°)")
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_camera_log_bars(ax: Any, values: Mapping[int, float], start: Mapping[int, float] | None,
                         every: Sequence[float], theme: str) -> None:
    """UX2-9: each camera's mean error at the shown step on the shared log axis (as Phase 3's
    solver bars), the start as grey ticks, so a 0.3 px bar is never a sliver."""
    t = es.stage_tokens(theme)
    scale = es.error_scale(theme)
    lo, hi = scale.log_range([x for x in every if np.isfinite(x) and x > 0])
    cams = sorted(values)
    pos = np.arange(len(cams))
    heights = [max(values[c], lo) for c in cams]
    ax.bar(pos, [h - lo for h in heights], bottom=lo, width=0.62, zorder=3,
           color=[es.camera_fill(c, ax) for c in cams])
    if start:
        ax.scatter([p for p, c in zip(pos, cams) if c in start],
                   [start[c] for c in cams if c in start], marker="_", s=260,
                   color=t["muted"], linewidths=2.0, zorder=4)
    for p, c in zip(pos, cams):
        ax.annotate(es.format_number(values[c]), (p, heights[p]), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5,
                    color=t["text"], gid="edu:value")
    scale.log_axis(ax, "y")
    ax.set_ylim(lo, hi * 1.6)
    ax.set_xlim(-0.6, len(cams) - 0.4)
    ax.set_ylabel(es.error_label("mean", short=True))
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    es._badge_ticks(ax, pos, cams, xlabel=None)  # noqa: SLF001 - foundation helper


def draw_face_pairs(ax: Any, pairs: Sequence[Any], theme: str, unit: str,
                    faces: Sequence[Any] = (), tag: str = "") -> None:
    """ED2-5 / SCI2-6: how much wider the solved cube is than designed across each pair of
    opposite faces (alignment-free), one bar per pair, tagged with both faces; a tick marks
    how the best turn + shift splits it between the two faces (that split is not unique)."""
    from matplotlib.transforms import blended_transform_factory
    t = es.stage_tokens(theme)
    factor = 1000.0 if unit == "µm" else 1.0
    offsets = {f.face: f.offset * factor for f in faces}
    vals = np.array([p.change * factor for p in pairs])
    ys = np.arange(len(pairs))
    ax.barh(ys, vals, height=0.56, color=t["accent"], alpha=0.85, zorder=3)
    ax.axvline(0, color=t["muted"], linewidth=1.0, zorder=2)
    for y, pair, v in zip(ys, pairs, vals):
        first = offsets.get(pair.faces[0])
        if first is not None and 0 < first < v:
            ax.plot([first, first], [y - 0.28, y + 0.28], color=t["stage"], linewidth=2.0,
                    zorder=4)
        text = f"{v:+.0f}" if unit == "µm" else f"{v:+.2f}"
        ax.annotate(f"{text} · {pair.angle_deg:.1f}° off parallel{tag}", (v, y), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.8,
                    color=t["text"])
    from matplotlib.transforms import offset_copy
    where = blended_transform_factory(ax.transAxes, ax.transData)
    ax.set_yticks(ys, [""] * len(ys))
    ax.tick_params(axis="y", length=0)
    for y, pair in zip(ys, pairs):
        for shift, face in ((-36, pair.faces[0]), (-14, pair.faces[1])):
            es.draw_face_tags(ax, [(0.0, y)], [face], size=8.5,
                              transform=offset_copy(where, fig=ax.figure, x=shift, y=0,
                                                    units="points"))
    ax.invert_yaxis()
    hi = max(float(vals.max()), 0.0) if vals.size else 1.0
    # UX5-6: the label "· 1.4° off parallel" needs its own width past the longest bar, with a
    # margin, so it stays inside the panel edge at L3 (the tagged label is the longest)
    ax.set_xlim(0, hi * (2.8 if tag else 2.1) or 1.0)
    ax.set_ylim(len(pairs) - 0.4, -0.6)
    ax.set_xlabel(f"wider across than designed ({unit})")
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def draw_coloured_hist(ax: Any, values: np.ndarray, clim: tuple[float, float], theme: str,
                       bins: int = 26) -> None:
    """Histogram whose bins carry the 3D view's colours (UX-33: the chart is the key)."""
    import matplotlib
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    hi = max(float(np.percentile(vals, 99)) * 1.05 if vals.size else 1.0, 1e-12)
    edges = np.linspace(0, hi, bins + 1)
    _n, edges, patches = ax.hist(np.clip(vals, 0, hi), bins=edges, zorder=3)
    cmap = matplotlib.colormaps[es.error_cmap(theme)]
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=clim[0], vmax=clim[1] or 1.0)
    for patch, left, right in zip(patches, edges[:-1], edges[1:]):
        patch.set_facecolor(cmap(norm((left + right) / 2)))
        patch.set_edgecolor("none")
    es.integer_axis(ax, "y")


def draw_signed_pairs(ax: Any, before: Mapping[int, float], after: Mapping[int, float],
                      theme: str) -> None:
    """Signed paired bars per camera (before hatched, after solid) around a zero line."""
    t = es.stage_tokens(theme)
    cams = sorted(set(before) | set(after))
    w = 0.36
    for k, cam in enumerate(cams):
        if cam in before:
            ax.bar(k - w / 2, before[cam], w, color=es.camera_colour(cam, ax), alpha=0.38,
                   hatch="///", edgecolor=es.camera_label_colour(ax, cam), linewidth=0.8,
                   zorder=3)
        if cam in after:
            ax.bar(k + w / 2, after[cam], w, zorder=3, **es.camera_mark_kwargs(cam, ax))
    ax.axhline(0, color=t["muted"], linewidth=1.0, zorder=2)
    vals = [v for v in list(before.values()) + list(after.values()) if np.isfinite(v)]
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - 0.12 * span, hi + 0.18 * span)
    ax.set_xlim(-0.6, len(cams) - 0.4)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    es._badge_ticks(ax, range(len(cams)), cams)  # noqa: SLF001 - foundation helper


def draw_log_pairs(ax: Any, before: Mapping[int, float], after: Mapping[int, float],
                   ylabel: str, unit: str) -> None:
    """Paired per-camera bars (before hatched, after solid) on a log axis, so 14.6 px and
    0.10 px are both readable (ED2-4: Phase 3's lens distortion next to Phase 4's)."""
    t = es._tokens_for(ax)  # noqa: SLF001 - foundation helper
    cams = sorted(set(before) | set(after))
    vals = [x for x in list(before.values()) + list(after.values()) if np.isfinite(x) and x > 0]
    ticks = [m * 10.0 ** e for e in range(-3, 4) for m in (1, 3)]
    lo = max([tk for tk in ticks if tk <= min(vals) / 1.2] or [ticks[0]]) if vals else 0.1
    hi = min([tk for tk in ticks if tk >= max(vals) * 1.3] or [ticks[-1]]) if vals else 10.0
    w = 0.38
    for k, cam in enumerate(cams):
        if cam in before and before[cam] > 0:
            ax.bar(k - w / 2, before[cam] - lo, w, bottom=lo, color=es.camera_fill(cam, ax),
                   alpha=0.38, hatch="///", edgecolor=es.camera_label_colour(ax, cam),
                   linewidth=0.8, zorder=3)
        if cam in after and after[cam] > 0:
            ax.bar(k + w / 2, after[cam] - lo, w, bottom=lo, zorder=3,
                   **es.camera_mark_kwargs(cam, ax))
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([tk for tk in ticks if lo <= tk <= hi]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _p: f"{val:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.6, len(cams) - 0.4)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_ylabel(f"{ylabel} ({unit}, log)" if unit else f"{ylabel}, log scale")
    es._badge_ticks(ax, range(len(cams)), cams, xlabel=None)  # noqa: SLF001


def cost_column_title(loss: str) -> str:
    """SCI6-1: the solver's cost is ½ Σ px² only for a linear loss; a robust loss (soft_l1 ...)
    minimises a different objective, so the column is named for what it holds."""
    if str(loss) == "linear":
        return "Phase 4 cost (½ Σ px²)"
    return f"Solver objective ({loss}, not px²)"


def cost_axis_label(loss: str) -> str:
    """The same rule for a chart axis (see :func:`cost_column_title`)."""
    if str(loss) == "linear":
        return "cost, ½ Σ squared error (px², log)"
    return f"solver objective ({loss}, log)"


def draw_held_band(ax: Any, theme: str, text: str) -> None:
    """ED6-1: the one visible 'held back' tag of a held-back result figure: a bordered band
    across the top of the figure, drawn in its own row so the layout reserves its space."""
    from matplotlib.patches import FancyBboxPatch
    t = es.stage_tokens(theme)
    tok = theme_tokens_for(theme)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.005, 0.1), 0.99, 0.8, boxstyle="square,pad=0",
                                facecolor=tok["surface_alt"], edgecolor=t["bad"], linewidth=1.0,
                                transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=10,
            fontweight="bold", color=t["bad"])


def draw_swap(ax: Any, before: Any, after: Any, swap: Mapping[str, float], theme: str) -> bool:
    """ED2-1 / ED3-4 / SCI3-1: the Phase 3 and Phase 4 errors with one part swapped back, as
    horizontal bars on the shared log error scale. The two results of the swap are the accent;
    the two real results are the accent, so no bar is coloured by its value (the worst is not
    the 'best' colour). Each swap puts one part back to its Phase 3 value and keeps the other at
    Phase 4's; the photo poses are re-fitted, nothing else re-solved (SCI7-1: one frame, no
    size fit, so both swaps and both real results are directly comparable)."""
    t = es.stage_tokens(theme)
    rows = [("Phase 3 result", before, "ref"),
            ("Lenses back to Phase 3", swap.get("lens_swap"), "swap"),
            ("Shape back to Phase 3", swap.get("shape_swap"), "swap"),
            ("Phase 4 result", after, "ref")]
    rows = [(label, float(v), kind) for label, v, kind in rows
            if v is not None and np.isfinite(v)]
    if len(rows) < 3:
        es.empty_panel(ax, "The swap test needs both saved states.")
        return False
    scale = es.error_scale(theme)
    lo, hi = scale.log_range([v for _l, v, _k in rows])
    ys = np.arange(len(rows))
    colours = [t["muted"] if kind == "swap" else t["accent"] for _l, _v, kind in rows]
    for y, (_label, v, kind), colour in zip(ys, rows, colours):
        ax.barh([y], [v - lo], left=lo, height=0.58, color=colour,
                alpha=0.55 if kind == "swap" else 0.95, zorder=3)
    for y, (_label, v, _kind) in zip(ys, rows):
        ax.annotate(es.format_number(v), (v, y), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, color=t["text"], fontweight="bold")
    scale.log_axis(ax, "x")
    ax.set_xlim(lo, hi * 3)
    ax.set_yticks(ys, [label for label, _v, _k in rows])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlabel(es.error_label("mean", short=True))
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return True


def draw_before_after_scatter(ax: Any, before_pairs: Any, after_pairs: Any, theme: str,
                              letter: str, *, verified: bool = True) -> None:
    """Each photo and camera: Phase 3 error against Phase 4 error (log-log), dashed no-change
    line. On a run that is not verified (``verified=False``) the count is neutral: photos moved,
    never "improved" (ED5-1)."""
    t = es.stage_tokens(theme)
    lo, hi = math.inf, 0.0
    better = total = 0
    for cam, (imgs_b, vals_b) in sorted((before_pairs or {}).items()):
        if cam not in after_pairs:
            continue
        imgs_a, vals_a = after_pairs[cam]
        common, ib, ia = np.intersect1d(imgs_b, imgs_a, return_indices=True)
        if not common.size:
            continue
        xb, ya = vals_b[ib], vals_a[ia]
        ax.scatter(xb, ya, s=20, alpha=0.85, zorder=3, **es.camera_mark_kwargs(cam, ax, "dot"))
        lo = min(lo, float(xb.min()), float(ya.min()))
        hi = max(hi, float(xb.max()), float(ya.max()))
        better += int(np.sum(ya < xb))
        total += int(common.size)
    if total:
        lo, hi = lo * 0.8, hi * 1.25
        ax.plot([lo, hi], [lo, hi], color=t["muted"], linestyle=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        _plain_log_ticks(ax, lo, hi)
    ax.set_xlabel("Phase 3 (px)")
    ax.set_ylabel("Phase 4 (px)")
    if verified:
        es.panel_title(ax, letter, "Each photo: before vs after",
                       f"{better} of {total} improved (below the dashes)")
    else:
        # SCI6-3: no tally on a held run. The fell/rose count sits next to a rising RMS and
        # would read as "almost every photo improved".
        es.panel_title(ax, letter, "Each photo (not verified)",
                       "one dot per photo; not a result")


def log_strip_pairs(ax: Any, before: Mapping[int, np.ndarray], after: Mapping[int, np.ndarray],
                    theme: str, *, seed: int = 7) -> None:
    """Per camera: before (pale, left) and after (solid, right) per-photo errors, log scale."""
    t = es.stage_tokens(theme)
    rng = np.random.default_rng(seed)
    cams = sorted(set(before) | set(after))
    for pos, cam in enumerate(cams):
        for offset, values, alpha, size in ((-0.2, before.get(cam), 0.4, 16),
                                            (0.2, after.get(cam), 0.9, 20)):
            if values is None or not len(values):
                continue
            v = np.asarray(values, dtype=float)
            v = v[np.isfinite(v) & (v > 0)]
            ax.scatter(pos + offset + rng.uniform(-0.13, 0.13, v.size), v, s=size, alpha=alpha,
                       zorder=3, **es.camera_mark_kwargs(cam, ax, "dot"))
            ax.plot([pos + offset - 0.16, pos + offset + 0.16], [np.median(v)] * 2,
                    color=t["text"], linewidth=1.6, zorder=4)
    every = np.concatenate([np.asarray(v, dtype=float) for v in (*before.values(), *after.values())
                            if v is not None and len(v)] or [np.array([1.0])])
    every = every[np.isfinite(every) & (every > 0)]
    if every.size:
        lo, hi = float(every.min()) * 0.8, float(every.max()) * 1.25
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        _plain_log_ticks(ax, lo, hi, "y")
    ax.set_xlim(-0.6, len(cams) - 0.4)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_ylabel(es.error_label("mean", per="photo", short=True))
    es._badge_ticks(ax, range(len(cams)), cams)  # noqa: SLF001 - foundation helper


# --------------------------------------------------------------------------------------
# The view


def _card(ax: Any, theme: str, edge: str | None = None) -> Any:
    """A rounded card filling ``ax`` (axes in 0-1 units), in the theme's card colours."""
    from matplotlib.patches import FancyBboxPatch
    tok = theme_tokens_for(theme)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    card = FancyBboxPatch((0.01, 0.03), 0.98, 0.94, boxstyle="round,pad=0,rounding_size=0.04",
                          facecolor=tok["surface_alt"], edgecolor=edge or tok["border"],
                          linewidth=1.2, transform=ax.transAxes, clip_on=False)
    ax.add_patch(card)
    return card


def _label_offset_pt(ax: Any, half_tick: float, x_range: float, y_range: float) -> float:
    """ED8-5: points from a ray's image-plane tick to its label: the tick's on-screen half
    length plus a margin, and never less than 26 pt. The axes is equal-aspect, so the
    smaller of the two scales sets the size of one data unit."""
    fig = ax.figure
    pos = ax.get_position()
    w_px = pos.width * fig.bbox.width
    h_px = pos.height * fig.bbox.height
    px_per_unit = min(w_px / x_range, h_px / y_range)
    return max(26.0, half_tick * px_per_unit * 72.0 / fig.dpi + 12.0)


def draw_corner_rays(ax: Any, summary: Any, theme: str, letter: str, held: bool = False) -> None:
    """ED3-2: a pixel fixes a ray; two cameras' rays for one corner cross at that corner.
    The two rays lie in one plane (two lines through a point always do), so the picture is
    exact in its angle; lengths along the rays are schematic."""
    t = es.stage_tokens(theme)
    ax.set_axis_off()
    example = getattr(summary, "example", None)
    if example is None:
        es.empty_panel(ax, "No corner was seen by two cameras in one photo, so no rays are drawn.")
        es.panel_title(ax, letter, "One pixel is a ray", None)
        return
    a, b = (np.asarray(d, dtype=float) for d in example.directions)
    cos = float(np.clip(a @ b, -1.0, 1.0))
    angle = math.degrees(math.acos(cos))
    ends = {0: np.array([1.0, 0.0]), 1: np.array([cos, math.sqrt(max(0.0, 1.0 - cos * cos))])}
    ax.set_xlim(-0.25, 1.45)
    ax.set_ylim(-0.45, 1.2)  # headroom: the top camera's centre sits near 1.12 (ED8-5)
    ax.set_aspect("equal")
    # ED8-5: the camera labels sit beyond the image-plane tick, whose on-screen half-length
    # grows with the figure, so the offset is the tick's half-length plus a margin.
    label_pt = _label_offset_pt(ax, half_tick=0.11, x_range=1.70, y_range=1.65)
    ax.scatter([0.0], [0.0], s=70, color=t["text"], zorder=6)
    ax.annotate("the corner", (0.0, 0.0), xytext=(-10, -14), textcoords="offset points",
                ha="right", va="top", fontsize=9, color=t["text"])
    for k in (0, 1):
        cam = example.cameras[k]
        end = 0.92 * ends[k]
        centre = 1.12 * ends[k]  # ED4-4: the camera centre, beyond the image plane
        ax.plot([0.0, centre[0]], [0.0, centre[1]], zorder=3, linewidth=2.2,
                **{key: val for key, val in es.camera_mark_kwargs(cam, ax, "line").items()
                   if key != "linewidth"})
        ax.scatter([centre[0]], [centre[1]], marker="s", s=44, zorder=7,
                   **es.camera_mark_kwargs(cam, ax, "dot"))
        across = np.array([-ends[k][1], ends[k][0]]) * 0.11
        ax.plot([end[0] - across[0], end[0] + across[0]], [end[1] - across[1], end[1] + across[1]],
                color=t["text"], linewidth=2.6, zorder=4)  # the camera's image plane
        u, v = example.pixels[k]
        # The label sits beside the ray at the image plane, on the side away from the wedge,
        # so it touches neither the ray, the image-plane tick nor the other ray.
        perp = np.array([-ends[k][1], ends[k][0]])
        if perp @ (ends[0] + ends[1]) > 0:
            perp = -perp
        ax.annotate(f"camera {es.camera_badge_label(cam)}\npixel ({u:.0f}, {v:.0f})", end,
                    xytext=tuple(label_pt * perp), textcoords="offset points",
                    ha="left" if perp[0] > 0.3 else "right" if perp[0] < -0.3 else "center",
                    va="bottom" if perp[1] > 0.3 else "top" if perp[1] < -0.3 else "center",
                    fontsize=9, color=t["text"],
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=t["stage"],
                              edgecolor="none", alpha=0.9))
    # ED4-4: the angle says why it matters: a wide angle pins the corner down, a narrow one
    # leaves it loose along the rays. It sits inside the wedge, clear of the labels.
    ax.annotate(f"{angle:.0f}° apart{' (not quoted)' if held else ''}\na wide angle pins the corner down",
                (0.78, 0.44 if held else 0.30), ha="center", va="center", fontsize=9,
                color=t["muted"],
                bbox=dict(boxstyle="round,pad=0.25", facecolor=t["stage"], edgecolor="none",
                          alpha=0.9))
    # ED7-5: on a held run the same picture stays (the idea is still taught) and is tagged as
    # showing no quoted precision; its numbers are the held run's, so none is read off it.
    if held:  # short enough for the narrower held layout at 1280 px
        sub = f"one corner, photo {example.photo + 1} · ■ camera centre · held back: not verified"
    else:
        sub = (f"one corner, photo {example.photo + 1}, two cameras · ■ camera centre · "
               "angles true, lengths schematic")
    es.panel_title(ax, letter, "One pixel is a ray (not quoted)" if held
                   else "One pixel is a ray; two rays meet at a corner", sub)


def draw_precision_bars(ax: Any, narrative: Any, visual: Any, um: float | None, theme: str,
                        letter: str) -> None:
    """ED3-2: the mean error as a length on the target, before (Phase 3) and after (Phase 4)."""
    t = es.stage_tokens(theme)
    if um is None:
        es.empty_panel(ax, "This run has no recorded units, so its pixels cannot be turned into µm.")
        es.panel_title(ax, letter, "The error on the target", None)
        return
    before = narrative.phase3_end_px()
    after = narrative.value("d43_final_px")
    rows = [("Phase 3 result", before, t["muted"]), ("Phase 4 result", after, t["accent"]),
            ("one pixel", 1.0, t["muted"])]
    rows = [(label, float(x) * um, colour) for label, x, colour in rows
            if x is not None and np.isfinite(x)]
    ys = np.arange(len(rows))
    ax.barh(ys, [x for _l, x, _c in rows], height=0.55, color=[c for _l, _x, c in rows],
            zorder=3)
    top = max(x for _l, x, _c in rows) if rows else 1.0
    for y, (_label, x, _c) in zip(ys, rows):
        ax.annotate(f"{x:.0f} µm", (x, y), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9.5, color=t["text"], fontweight="bold")
    ax.set_yticks(ys, [label for label, _x, _c in rows])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.35)
    ax.set_xlabel("µm on the target")
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    es.panel_title(ax, letter, "What the error is on the target",
                   f"{um:.0f} µm per pixel in this run's cameras (median)")


def draw_redo_card(ax: Any, visual: Any, theme: str, letter: str) -> None:
    """ED3-2 L2: when this calibration stops being right."""
    t = es.stage_tokens(theme)
    _card(ax, theme)
    # ED4-9: one trigger per line, so the list scans; each line is its own text, since
    # card_text re-wraps whitespace and would merge the lines again.
    ax.text(0.04, 0.88, "Calibrate again after any of these:", transform=ax.transAxes,
            ha="left", va="top", fontsize=10.5, color=t["text"], fontweight="bold")
    triggers = ["a lens is refocused, or its aperture or zoom is changed",
                "a camera or the target is moved",
                "the working distance or the field of view changes"]
    for k, trigger in enumerate(triggers):
        ax.text(0.06, 0.88 - 0.16 * (k + 1), f"•  {trigger}", transform=ax.transAxes, ha="left",
                va="top", fontsize=10.5, color=t["text"])
    # UX7-2: a little more room below the last line, so the card's bottom padding matches its top
    ax.text(0.04, 0.88 - 0.16 * (len(triggers) + 1) - 0.02,
            "Then run Phases 1 to 4 and check the error on the target before you measure.",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5, color=t["muted"])
    es.panel_title(ax, letter, "Before you measure again",
                   "what makes this calibration stop being right")


def _hanging_card(ax: Any, x: float, y_centre: float,
                  paragraphs: Sequence[tuple[str, str]], *, wrap: int, colour: str, edge: str,
                  fill: str, fontsize: float = 11.0) -> None:
    """Round 9 (UX8-2 / UX7-1): a card of labelled paragraphs in axes coordinates, laid out at
    draw time by offset boxes (so the text and the card are sized with the final fonts). Each
    marker ("✕  ", "To do: ") sits at the card's left edge and its wrapped body is packed beside
    it, top-aligned, so every continuation line hangs under the first line's text. The card
    is only as wide as its text and is centred on ``y_centre``."""
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
    rows = []
    for marker, body in paragraphs:
        lines = textwrap.wrap(body, width=wrap)
        head = TextArea(marker, textprops=dict(fontsize=fontsize, color=colour))
        text = TextArea("\n".join(lines), textprops=dict(fontsize=fontsize, color=colour,
                                                          linespacing=1.5))
        rows.append(HPacker(children=[head, text], align="top", pad=0, sep=0))
    block = VPacker(children=rows, align="left", pad=0, sep=9)
    card = AnchoredOffsetbox(loc="center left", child=block, pad=0.8, borderpad=0.0,
                             frameon=True, bbox_to_anchor=(x - 0.012, y_centre),
                             bbox_transform=ax.transAxes)
    card.patch.set_boxstyle("round,pad=0,rounding_size=0.4")
    card.patch.set_facecolor(fill)
    card.patch.set_edgecolor(edge)
    card.patch.set_linewidth(1.4)
    ax.add_artist(card)


def draw_closing_held(ax: Any, theme: str, reasons: Sequence[Any] = (), n_excluded: int = 0,
                      letter: str = "A", wrap: int = 100) -> None:
    """The closing stage for a run that did not pass its gate: no precision is quoted; the
    gate's reasons are shown in plain words with what to do (UX3-4), as cards, and the card
    underneath says what this stage shows once a run passes."""
    from matplotlib.patches import FancyBboxPatch
    t = es.stage_tokens(theme)
    tok = theme_tokens_for(theme)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    items = [plain_blocking(r, n_excluded) for r in list(reasons)[:3]] or [
        ("The quality gate did not pass.",
         "Press ‘Phase 4 - Self-Calibration ▶’ again in Phase 3, then check the Can we trust it? stage.")]
    # UX6-2: a held run shows only its own reasons and what to do, as cards that fill the
    # panel, and one line saying what is withheld. The generic four-step chain is left out:
    # it is the same on every run, so on a held run it reads as this run's result.
    # UX7-1: one reason leaves room under its card for a neutral card saying what a passing
    # run shows here (the reason cards keep the top; the note line stays at the bottom).
    passing = len(items) == 1
    gap, top0, bottom0 = 0.03, 0.95, (0.56 if passing else 0.17)
    avail = top0 - bottom0
    # ED7-5: one reason gets a card tall enough for its wrapped lines (the text sits inside it)
    step = min(0.45 if len(items) == 1 else 0.30, (avail - gap * (len(items) - 1)) / len(items))
    block = len(items) * step + gap * (len(items) - 1)
    top_of_block = top0 - (avail - block) / 2  # the cards sit centred above the note line
    for i, (what, todo) in enumerate(items):
        top = top_of_block - i * (step + gap)
        # The reason on one line and what to do on the next: an explicit break, so no word is
        # left alone on a wrapped last line (UX6-2). Round 9 (UX8-2 / UX7-1): the continuation
        # lines hang under the text after the marker, and the card is sized to its text.
        _hanging_card(ax, 0.04, top - step / 2, [("✕  ", what), ("To do: ", todo)],
                      wrap=wrap, colour=t["text"], edge=t["bad"], fill=tok["surface_alt"])
    if passing:
        _hanging_card(ax, 0.04, 0.30,
                      [("", "When a run passes, this stage shows: A one pixel as a ray; B the error "
                            "on the target in µm; C what to check before you measure again.")],
                      wrap=wrap, colour=t["text"], edge=t["muted"], fill=tok["surface"])
    ax.text(0.01, 0.04, textwrap.fill("No precision is quoted from this run, so no µm figure is "
                                      "shown until a run passes its gate.", width=wrap),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color=t["muted"])
    es.panel_title(ax, letter, "No precision to quote yet",
                   "what the gate found, and what to do next")


class Phase4View(QWidget):
    """Phase 4's five visual stages for one saved run (see the module docstring).

    The view opens at once. The derivation, the replay and the visual data are prepared on a
    worker thread (``_prepare_data``); until they are ready each stage shows a 'Preparing the
    walkthrough' state, and the stages fill in when the data arrives. ``narrative``, ``visual``,
    ``playback`` and ``_derivation`` are None until then.
    """

    def __init__(self, context: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self.run_context = self._run_context(context)
        self.metadata = plain(context.metadata)
        self._state: Any = None
        self._trace: Any = None
        self._derivation: Any = None
        self.narrative: Phase4Narrative | None = None
        self.visual: Phase4Visual | None = None
        self.playback: Any = None
        self._prepare_error: str | None = None
        self._prepare_status = "Preparing the walkthrough ..."
        self._precision_checked = False  # the Jacobian summary is computed once, on demand

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.stage_view = EducationStageView(STAGES, self, phase_label="Phase 4")
        self.stage_view.setAccessibleName(
            f"Phase 4 explanation · run {context.run_id} · {context.status}")
        layout.addWidget(self.stage_view)
        self.split = SplitVisual(self.stage_view.visual)
        self.morph: MorphBar | None = None  # built with the data: it needs the exaggeration
        self.player: ScenePlayer | None = None
        self._play_position = 0.0
        self._morph_items: dict[str, Any] = {}
        self._morph_phase = -1
        self._showing = -1
        self._worker = EducationWorker(self)
        self._worker.ready.connect(self._on_ready)
        self._worker.failed.connect(self._prepare_failed)
        self._worker.status.connect(self._prepare_progress)
        self.stage_view.stageChanged.connect(self._show_stage)
        self.stage_view.levelChanged.connect(lambda _level: self._show_stage(self.current_stage()))
        self.stage_view.themeChanged.connect(lambda _name: self._show_stage(self.current_stage()))
        self._worker.submit("phase4", self._prepare_data)
        self._show_stage(0)

    # -- preparation: _prepare_data runs on the worker thread, the rest on the GUI thread ----

    def is_ready(self) -> bool:
        """True once the stage data is prepared (the stages can be drawn)."""
        return self.visual is not None

    def _prepare_data(self, report: Callable[[str], None]) -> tuple[Any, ...]:
        """Everything the stages need, computed without Qt (runs on the worker thread)."""
        report("Reading the saved run ...")
        state = self.run_context.phase3_state_source()
        trace = read_trace(self.context.run_dir, self.context.run_id, "phase4")
        report("Recalculating the saved result. This can take a few seconds ...")
        derivation = derive_phase4(self.run_context, include_playback=True)
        report("Preparing the stages ...")
        narrative = Phase4Narrative(self.run_context, self.metadata, state, trace, derivation,
                                    str(self.context.run_id))
        visual = prepare_phase4_visual(self.run_context, self.metadata, state, derivation, trace)
        return state, trace, derivation, narrative, visual, visual.playback()

    def _on_ready(self, key: Any, data: Any) -> None:
        if key == "precision":
            self._precision_checked = True
            self._show_stage(self.current_stage())
        elif key == "swap":
            self._show_stage(self.current_stage())
        else:
            self._prepared(key, data)

    def _needs_precision(self, index: int) -> bool:
        """The corner precision (the Jacobian summary) is shown at level 3 and on the trust
        stage, so it is worked out before those stages draw, not when the view opens."""
        return ((index == 3 or self.level() >= 3) and self.visual is not None
                and not self._precision_checked and self.visual._precision is None)

    def _needs_swap(self, index: int) -> bool:
        """The swap test is worked out on the Result stage's first visit (a few seconds on a
        cold cache), so it runs on the worker first. ``_swap_source`` is set until it runs."""
        return (index == 4 and self.visual is not None
                and getattr(self.visual, "_swap_source", None) is not None)

    def _await_background(self, index: int) -> None:
        if self._needs_precision(index):
            self._show_pending(index, "Working out how precise the corners are ...")
            if not self._worker.is_busy():  # a job already running delivers to the same stage
                self._worker.submit("precision", self._compute_precision)
        else:
            self._show_pending(index, "Running the swap check on the lens and the shape ...")
            if not self._worker.is_busy():
                self._worker.submit("swap", self._compute_swap)

    def _compute_precision(self, _report: Callable[[str], None]) -> bool:
        """Runs on the worker thread; fills the visual's precision cache."""
        self.visual.jacobian_summary()
        return True

    def _compute_swap(self, _report: Callable[[str], None]) -> bool:
        """Runs on the worker thread; fills the visual's swap result (``swap_px``)."""
        self.visual.swap_results()
        return True

    def _prepared(self, _key: Any, data: tuple[Any, ...]) -> None:
        (self._state, self._trace, self._derivation, self.narrative,
         self.visual, self.playback) = data
        self.morph = MorphBar(self._morph_changed, self._exaggeration_choices(),
                              self.visual.exaggeration, self._morph_describe, self._morph_short,
                              self)
        self.morph.hide()
        self._showing = -1
        self._show_stage(self.current_stage())

    def _prepare_failed(self, key: Any, message: str) -> None:
        if key in ("precision", "swap"):  # context checks: the stage still shows what it can
            self._precision_checked = self._precision_checked or key == "precision"
            self._show_stage(self.current_stage())
            return
        self._prepare_error = message
        self._show_stage(self.current_stage())

    def _prepare_progress(self, _key: Any, text: str) -> None:
        self._prepare_status = text
        if not self.is_ready() and self._prepare_error is None and self._showing >= 0:
            self._show_pending(self._showing)

    # -- plumbing -------------------------------------------------------------------------

    @staticmethod
    def _run_context(context: Any) -> Any:
        """Use the shared immutable context or rebuild it from the selected run."""
        if callable(getattr(context, "phase3_state_source", None)):
            return context
        from pyCamSet.workflow.education_trace import RunContext
        metadata = plain(context.metadata)
        return RunContext.from_run(
            context.manager, {**metadata, "_run_dir": context.run_dir}, context.phase)

    def current_stage(self) -> int:
        return self.stage_view.current_stage()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Block until the stages are prepared (tests and screenshot tools only)."""
        self._worker.wait_until_idle(timeout)
        return self.is_ready()

    def close_background(self) -> None:
        """Cancel any preparation still running (called when the dialog closes)."""
        self._worker.close()

    def level(self) -> int:
        return self.stage_view.level()

    def set_level(self, level: int) -> None:
        self.stage_view.set_level(level)

    def set_stage(self, index: int) -> None:
        self.stage_view.set_current_stage(index)

    def step_back(self) -> None:
        """Previous stage (shared storyboard seam)."""
        self.stage_view.stepper.previous()

    def step_forward(self) -> None:
        """Next stage (shared storyboard seam)."""
        self.stage_view.stepper.next()

    def stop_for_context_change(self) -> None:
        self.stage_view.stop_for_context_change()
        if self.morph is not None:
            self.morph.stop_for_context_change()

    @property
    def exaggeration(self) -> int:
        """The display exaggeration of the shape change (stage 2's ⚙ choice, reused in 3)."""
        return self.morph.exaggeration if self.morph is not None else 1

    @property
    def step_controls(self) -> Any:
        """The recorded-step controls while 'Watching the solver' plays a trace, else None."""
        if (self._showing == 2 and self.player is not None and self.playback is not None
                and self.playback.playable):
            return self.player.controls
        return None

    def stage_text(self, index: int | None = None, level: int | None = None) -> str:
        """Everything a learner can read on a stage (moving there first when ``index`` or
        ``level`` is given): caption, chips, the 3D panel's titles and the Numbers drawer.

        Also the stage's text alternative for screen readers and the tests."""
        if level is not None:
            self.set_level(level)
        if index is not None:
            self.set_stage(index)
        parts = [self.stage_view.caption.text()]
        parts += [f"{chip.value} {chip.label}" for chip in self.stage_view.chips.chips()]
        if self.stage_view.visual.currentWidget() is self.split:
            parts += [self.split.title.text(), self.split.subtitle.text()]
        table = self.stage_view.numbers.table
        for row in range(table.rowCount()):
            cells = [table.item(row, col).text() for col in range(table.columnCount())
                     if table.item(row, col) is not None]
            parts.append(" | ".join(cells))
        return "\n".join(part for part in parts if part)

    def numbers_rows(self) -> list[tuple[str, ...]]:
        """The current stage's Numbers drawer rows, as shown."""
        table = self.stage_view.numbers.table
        return [tuple(table.item(row, col).text() if table.item(row, col) is not None else ""
                      for col in range(table.columnCount())) for row in range(table.rowCount())]

    def _theme(self) -> str:
        return self.stage_view.theme()

    def _caption(self, sentences: Sequence[str]) -> str:
        """Captions: <= 2 / 3 / 4 sentences at L1 / L2 / L3.

        A run warning (failed, unconfirmed start, held back) comes first. At L1 it takes one of
        the two sentences; from L2 it is added on top, so the level's own explanation is never
        pushed out by it (the shared contract counts it: the body is cut to fit).
        SCI7-3: on a held run every sentence that shows a measured number carries the tag.
        """
        if self._held():
            sentences = [_held_caption(x) for x in sentences]
        level = self.level()
        warning = self.narrative.headline_warning(on_trust_stage=self._showing == 3)
        limit = {1: 2, 2: 3, 3: 4}[level]
        body = [s for s in sentences if s]
        if not warning:
            return " ".join(body[:limit])
        return " ".join([warning] + body[:limit - 1])

    def _show_stage(self, index: int) -> None:
        if not self.is_ready():
            self._show_pending(index)
            return
        if self._needs_precision(index) or self._needs_swap(index):
            self._await_background(index)
            return
        if self.player is not None and self._showing == 2:
            self._play_position = self.player.position
        self.stage_view.set_busy(None)
        self.stop_for_context_change()
        self._showing = index
        # Each stage's Terms list is found from its own caption and chips unless the stage sets
        # one (stage 4 does); reset it here so a list set for one stage never leaks into another.
        self.stage_view.set_glossary(None)
        stage = (self._stage_start, self._stage_shape, self._stage_solver, self._stage_trust,
                 self._stage_result, self._stage_closing)[max(0, min(index, 5))]
        stage()

    def _show_pending(self, index: int, status: str | None = None) -> None:
        """The stage frame while data is prepared: a title, the stepper and this text."""
        self.stop_for_context_change()
        self._showing = index
        self.stage_view.set_glossary(None)
        if self._prepare_error is not None:
            title = "The walkthrough could not be prepared"
            message = (f"The walkthrough could not be prepared. Not shown: "
                       f"{self._prepare_error}.")
        else:
            title = "Preparing the walkthrough"
            message = f"{status or self._prepare_status} The stages fill in when this is ready."
        self.stage_view.setAccessibleDescription(message)
        self.stage_view.set_chips([])
        self.stage_view.set_caption(message)
        self.stage_view.set_controls(None)
        self.stage_view.set_numbers((), ())
        self.stage_view.set_busy(None if self._prepare_error is not None
                                 else (status or self._prepare_status).rstrip(" ."))

        def draw(fig: Any, _theme: str) -> None:
            ax = fig.add_subplot()
            es.empty_panel(ax, message)
            es.panel_title(ax, None, title, "the stages fill in when the data is ready")
        self.stage_view.draw_figure(draw)

    def _use_split(self) -> SplitVisual:
        self.split.set_theme(self._theme())
        self.stage_view.set_visual(self.split)
        return self.split

    def _set_chips(self, chips: Sequence[Chip]) -> None:
        """Show chips at their full size; the caption wraps instead of the chips overlapping.
        SCI7-3: on a held run a chip that shows a measured value says so in its label."""
        if self._held():
            # short tag on chips: a long label would narrow the caption beside the chip row
            chips = [dataclasses.replace(c, label=f"{c.label} (not verified)")
                     if _measures(c.value) and "not verified" not in c.label else c
                     for c in chips]
        self.stage_view.set_chips(list(chips))
        self._fit_chips()
        QTimer.singleShot(0, self._fit_chips)  # again once the chip fonts are polished

    def _fit_chips(self) -> None:
        """Pin the chip row to its polished size, keeping it to about half the band so the
        caption stays readable (trailing chips go first; their numbers stay in the drawer)."""
        row = self.stage_view.chips
        while True:
            layout = row.layout()
            for child in row.findChildren(QWidget):
                # StatChips.set_chips deleteLater()s the old chips without hiding them; until
                # the deferred delete runs they would still paint under the new ones.
                if (child.objectName() == "eduChip" and layout is not None
                        and layout.indexOf(child) < 0):
                    child.hide()
                child.ensurePolished()
            row.ensurePolished()
            if layout is not None:
                layout.invalidate()
                layout.activate()
            chips = row.chips()
            width = self.stage_view.width()
            if width > 400 and len(chips) > 1 and row.sizeHint().width() > 0.5 * width:
                self.stage_view.set_chips(chips[:-1])
                continue
            row.setMinimumWidth(row.sizeHint().width())
            return

    def _numbers(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        """SCI7-3: on a held run every row that shows a measured value carries the tag, in its
        last cell, so no Numbers row reads as a result."""
        held = self._held()
        out = []
        for row in rows:
            cells = [str(c) for c in row]
            if held and cells and any(_measures(c) or _BARE_NUMBER.match(c) for c in cells):
                joined = " | ".join(cells)
                if HELD_TAG not in joined:
                    cells[-1] = f"{cells[-1]} · {HELD_TAG}" if cells[-1] else HELD_TAG
            out.append(tuple(cells))
        self.stage_view.set_numbers(headers, out)

    def _unavailable_figure(self, title: str, message: str) -> None:
        def draw(fig: Any, _theme: str) -> None:
            ax = fig.add_subplot()
            es.empty_panel(ax, message)
            es.panel_title(ax, None, title, "nothing is drawn that this run cannot back")
        self.stage_view.set_controls(None)
        self.stage_view.draw_figure(draw)

    def _exaggeration_choices(self) -> list[int]:
        auto = self.visual.exaggeration
        choices = sorted({1, auto, *[f for f in EXAGGERATIONS
                                     if auto / 10 <= f <= auto * 5]})
        return choices

    def _length_unit(self) -> str:
        """µm or mm for lengths on this stage (from the corrections' size)."""
        _sizes, unit = self.visual.correction_display()
        return unit if unit in ("µm", "mm") else "mm"

    # -- scene building -----------------------------------------------------------------

    def _display_frame(self) -> np.ndarray:
        """The 3D views draw in the frame Phase 3 ended in (ED2-7 / UX2-4): the target at the
        reference photo, the world's +z up, framed by the same ``fit_view`` call as Phase 3's
        rig, so the rig does not seem to change between the phases. (The reference photo's
        pose is the solver's fixed identity in both phases.)"""
        return np.eye(3)

    def _rig_up(self) -> np.ndarray:
        """Rotation whose +z is the rig's up (the cameras' mean image-up): only used to pick
        which cube axis a 2D slice looks down."""
        cams = self.visual.start.cameras if self.visual.start is not None else (
            self.visual.final.cameras if self.visual.final is not None else ())
        return display_rotation(cams)

    def _add_cameras(self, scene: Any, Q: np.ndarray, cameras: Sequence[CameraPose],
                     labels: Mapping[int, str] | None = None, prefix: str = "cam") -> None:
        scale = 0.30 * self.visual.target_size  # Phase 3's glyph size for a telecentric rig
        for cam in cameras:
            R, t = _camera_display(Q, cam)
            scene.add_camera(f"{prefix}{cam.index}", cam.index, R, t, scale=scale,
                             telecentric=cam.telecentric,
                             label=(labels or {}).get(cam.index))

    def _add_outline(self, scene: Any, Q: np.ndarray) -> None:
        """The printed design's bounding box as faint lines (context for the corners)."""
        if self.visual.printed is None:
            return
        origins, ends = box_edges(self.visual.printed)
        scene.add_rays(_to_display(Q, origins), _to_display(Q, ends), name="outline",
                       colour=es.stage_tokens(self._theme())["muted"], width=1.4, opacity=0.8)

    def _view(self, scene: Any) -> None:
        """Frame the whole scene with the shared auto-framing (UX-10): the view that separates
        the cameras best, every badge inside the frame, a gutter for the colour bar."""
        try:
            if "cam0" in scene.names() or not self._rig_view():
                scene.fit_view()  # Phase 3's rig call (default margin): the same viewpoint
            else:  # a stage without cameras still looks from where the rig is seen
                az, el = self._rig_view()
                scene.fit_view(azimuth=az, elevation=el, margin=0.22)
        except Exception:  # noqa: BLE001 - a framing failure must not lose the stage
            scene.set_view(azimuth=-58.0, elevation=24.0, up=(0, 0, 1))

    def _rig_view(self) -> tuple[float, float] | None:
        """(azimuth, elevation) ``fit_view`` picks for the rig (stage 1's view), for the
        stages that draw the target without its cameras."""
        if "_rig_view_cache" not in self.__dict__:
            from pyCamSet.gui.education_scene3d import choose_view
            v = self.visual
            cams = (v.start or v.final).cameras if (v.start or v.final) is not None else ()
            view = None
            if cams and v.printed is not None:
                pts = v.printed[np.all(np.isfinite(v.printed), axis=1)]
                lo, hi = pts.min(axis=0), pts.max(axis=0)
                view = choose_view(np.array([c.centre for c in cams]), (lo + hi) / 2,
                                   glyph_size=0.30 * v.target_size,
                                   target_radius=float(np.linalg.norm(hi - lo)) / 2)
            self._rig_view_cache = view
        return self._rig_view_cache

    def _telecentric_note(self) -> str:
        if not any(self.visual.telecentric):
            return "Cameras are drawn at their solved positions."
        return ("Telecentric cameras are drawn at a fixed distance; their model also fits a "
                "depth-dependent magnification and a shift of the image centre, which are not drawn.")

    # -- stage 1 -------------------------------------------------------------------------

    def _stage_start(self) -> None:
        v, n = self.visual, self.narrative
        level = self.level()
        start_px = (v.start.mean_px if v.start is not None
                    else (self.metadata.get("diagnostics") or {}).get("D4.3_initial_euclid_px"))
        posed = v.n_images - len(v.excluded) if v.n_images else None
        excl = len(v.excluded)
        single = bool(v.excluded) and all(len(e.cameras) == 1 for e in v.excluded)
        chips = [Chip(f"{_px(start_px)} px", "mean error at the start",
                      tone=None if n.trusted_start() else "warn"),
                 Chip(f"{posed} / {v.n_images}" if posed is not None else "not recorded",
                      "photos placed", tone="warn" if excl else "good"),
                 Chip(str(v.n_cameras or "—"), "cameras")]
        self._set_chips(chips)
        caption = [
            ("Phase 4 starts from Phase 3's cameras and target; each corner is coloured by how "
             "far off Phase 3 still was."),
            (f"{_left_out_label([e.image for e in v.excluded])} "
             f"{'is' if excl == 1 else 'are'} left out: no target position could be found"
             + (" (each was seen by one camera only)." if single else ".")
             if excl else "Every photo had a target position, so all of them take part."),
            ("Corners that stay off in every photo hint that the target is not exactly its "
             "drawing; Phase 4 tests that."),
            self._telecentric_note(),
        ]
        self.stage_view.set_caption(self._caption(caption))
        self._numbers_start()
        if not v.available or v.start is None or v.start_errors is None:
            reason = v.reason or v.errors_reason or "the start could not be replayed"
            self._unavailable_figure("Start from Phase 3", f"Not shown: {reason}.")
            return
        split = self._use_split()
        self.stage_view.set_controls(None)
        scene = split.scene
        scene.clear()
        Q = self._display_frame()
        per_cam = v.phase3_camera_mean or v.start.per_camera
        # As Phase 3's last rig (ED2-7): each camera's mean error in brackets, same viewpoint.
        self._add_cameras(scene, Q, v.start.cameras,
                          labels={c: f"({es.format_px(x)})" for c, x in per_cam.items()})
        per_point = v.start_errors.per_point()
        pts = _to_display(Q, v.printed)
        seen = np.isfinite(per_point)
        scale = es.error_scale(self._theme())
        scene.add_points(pts[seen], per_point[seen], name="corners", scalar_bar="px", size=9,
                         **scale.scene_kwargs())
        if (~seen).any():
            scene.add_points(pts[~seen], name="unseen", colour=es.stage_tokens(self._theme())["muted"],
                             size=4)
        self._add_outline(scene, Q)
        self._view(scene)
        split.set_titles(
            "A  The rig and target as Phase 4 started",
            "Phase 3's result, seen as Phase 3 ended · colour: corner error · (px): camera mean")
        by_cam = {c: vals for c, (_imgs, vals) in v.start_errors.by_camera_image().items()}
        worst = None
        if level >= 2:
            best = max(((c, i, val) for c, vals in by_cam.items() for i, val in enumerate(vals)),
                       key=lambda x: x[2], default=None)
            worst = (best[0], best[1]) if best else None

        def side(fig: Any, theme: str) -> None:
            rows = [3.0, 1.15] + ([1.9] if level >= 3 else [])
            gs = fig.add_gridspec(len(rows), 1, height_ratios=rows)
            ax = fig.add_subplot(gs[0])
            es.strip_plot_by_camera(ax, by_cam, highlight=worst, unit="px",
                                    ylabel=es.error_label("mean", per="photo", unit=False,
                                                          short=True))
            if level >= 2:
                t = es.stage_tokens(theme)
                for pos, cam in enumerate(by_cam):
                    if cam in per_cam:
                        ax.plot([pos - 0.34, pos + 0.34], [per_cam[cam]] * 2, color=t["text"],
                                linewidth=2.2, solid_capstyle="butt", zorder=5)
            es.panel_title(ax, "B", "Error per photo, by camera",
                           "one dot per photo" + (" · ring = worst photo · bar = camera mean"
                                                  if worst else ""))
            bx = fig.add_subplot(gs[1])
            draw_excluded_cards(bx, v.excluded, theme, detail=level >= 2)
            es.panel_title(bx, "C", "Left out: no target position",
                           "photo number · the camera that saw it" if v.excluded else None)
            if level >= 3:
                cx = fig.add_subplot(gs[2])
                vals = per_point[np.isfinite(per_point)]
                t = es.stage_tokens(theme)
                cx.hist(vals, bins=30, color=t["accent"], alpha=0.85)
                cx.axvline(np.median(vals), color=t["text"], linestyle=(0, (4, 3)), linewidth=1.1)
                cx.set_xlabel("mean error of one corner over all photos (px)")
                cx.set_ylabel("corners")
                es.integer_axis(cx, "y")
                es.panel_title(cx, "D", "How far off each corner is",
                               f"median {np.median(vals):.2f} px over {vals.size} corners"
                               + ("" if self._usable() else " (not verified)"))
        split.set_side(side, (3, 2))

    def _numbers_start(self) -> None:
        n, state = self.narrative, self._state
        if self.run_context.status == "failed":
            self._numbers(["Item", "Value", "Evidence"],
                          [("Phase 4 result", "failed", n.status_copy())])
            return
        # The used camset's path is named only once the start is confirmed: an unconfirmed
        # (e.g. foreign) start must not be credited to, or name, another folder.
        used = (str(state.path) if state.path is not None and state.confirmed
                else "not confirmed" if state.path is not None else "not available")
        rows = [("Phase 3 camset used", used, state.verification.value),
                ("Phase 3 ancestry", state.origin_run_id or "not traceable", state.source_check),
                ("Phase 3 ended at", f"{fmt(n.phase3_end_px())} px", "Phase 3's final mean error (baseline)"),
                ("Phase 4 started at",
                 f"{fmt((self.metadata.get('diagnostics') or {}).get('D4.3_initial_euclid_px'))} px",
                 ("D4.3 initial mean error" if self.level() >= 2 else "mean error where Phase 4 began")
                 + (f"; {self.visual.start_label}" if self.visual.start_label else "")),
                ("Reference photo (fixed pose)",
                 {True: "same in both phases", False: "different"}.get(state.fixed_pose_matches, "not checked"),
                 {"fixed_pose": "different reference photo",
                  "fixed_parameters": "different fixed parameters"}.get(state.start_mismatch, "no mismatch")),
                ("Reference photo", "; ".join(state.photo_names) or "cannot be named",
                 state.source_check),
                ("Effective fixed parameters",
                 ", ".join(state.effective_fixed_parameter_differences) or "none recorded",
                 "different" if state.start_mismatch == "fixed_parameters" else "no effective difference")]
        for ex in self.visual.excluded:
            rows.append((f"{es.photo_label(ex.image)} left out",
                         ", ".join(f"camera {c + 1}" for c in ex.cameras) or "no camera",
                         f"{ex.total} corners; no usable target pose; index (0-based) {ex.image}"))
        rows += [("Status", self.run_context.status, n.status_copy()),
                 ("Where it started", "", n.input_copy()),
                 ("Start settings", "", n.start_copy()),
                 ("Reference photo and position frame", "", n.gauge_copy())]
        if self.level() >= 2 and n.gauge_detail():
            rows.append(("Position frame", "not checked", n.gauge_detail()))
        rows += n.rig_comparison_rows()
        self._numbers(["Item", "Value", "Evidence"], rows)

    # -- stage 2 -------------------------------------------------------------------------

    def _morph_phase_of(self, position: int) -> int:
        """0 printed, 1 faces moving, 2 the change inside the faces being added."""
        if position <= 0:
            return 0
        return 1 if position <= MORPH_HALF or self.visual.plate is None else 2

    def _morph_describe(self, position: int) -> str:
        if self.visual.plate is None:
            if position == 0:
                return "Printed design: every corner where the drawing puts it."
            if position >= MORPH_POSITIONS - 1:
                return "Refined shape (corrections exaggerated, display only)."
            return f"{round(100 * position / (MORPH_POSITIONS - 1))} % of the way to the refined shape."
        if position == 0:
            return "Printed design: every corner where the drawing puts it."
        if position < MORPH_HALF:
            return f"Faces moving as rigid plates: {10 * position} % of their move."
        if position == MORPH_HALF:
            return "Each face moved and tilted as one rigid plate."
        if position < MORPH_POSITIONS - 1:
            return (f"Adding the change inside each face: {10 * (position - MORPH_HALF)} %.")
        return "Refined shape: the face moves plus the change inside each face."

    def _morph_short(self, position: int) -> str:
        if position == 0:
            return f"step 1 of {MORPH_POSITIONS} · printed"
        if position >= MORPH_POSITIONS - 1:
            return f"step {MORPH_POSITIONS} of {MORPH_POSITIONS} · refined"
        # UX2-21: the counter slot reads like every other bar's ("step 11 of 21 · ...").
        step = f"step {position + 1} of {MORPH_POSITIONS}"
        if self.visual.plate is None:
            return step
        return f"{step} · " + ("faces moved" if position == MORPH_HALF else
                               "faces moving" if position < MORPH_HALF else "inside added")

    def _shape_caption(self, position: int) -> list[str]:
        v = self.visual
        unit = self._length_unit()
        factor = self.exaggeration
        summary = v.face_summary()
        lead = ("A hand-glued cube is never exactly its drawing, so every corner may move."
                if summary else
                "In Phase 4 every corner of the target may move: the solver refines the "
                "target's own shape together with the cameras.")
        phase = self._morph_phase_of(position)
        if summary is None:
            second = (f"Each line runs from a corner's printed position to where it is now, drawn "
                      f"×{factor} longer; colour shows how far it has moved.")
        elif phase == 0:
            second = ("Press Play: each face moves as one plate; the change inside follows.")
        elif phase == 1:
            second = (f"Each face moves as one plate: "
                      f"{_length(summary['offset_min'], unit)}–{_length(summary['offset_max'], unit)} "
                      f"further out than drawn, tilted up to {summary['tilt_max']:.1f}°.")
        else:
            second = (f"Then the change inside each face{' is added' if self._usable() else ''}: "
                      f"{_length(summary['inside_mean'], unit)} on average, small next to the "
                      "face moves.")
        widths = self._width_range(unit)
        if summary is not None and widths and phase >= 1:
            # ED2-5 / SCI2-6: say plainly that the cube came out wider than drawn while the
            # printed squares keep their size, and offer a cause, labelled as a guess.
            # One sentence only: _caption cuts the list by item, not by sentence (L2 limit 3).
            if self._usable():
                third = (f"The cube is {widths} wider than drawn, though its squares fix the scale; "
                         "a guess for the cause: paper, glue or a larger body.")
            else:  # a held run states the number only (the tag follows it); no guessed cause
                third = f"The cube is {widths} wider than drawn."
        else:
            third = ("The printed squares keep their size, which fixes the scale; faces may sit "
                     "further in or out than drawn.")
        fourth = self._squareness_sentence()
        return [lead, second, third, fourth]

    def _width_range(self, unit: str, compact: bool = False) -> str:
        """"+423 to +511 µm": how much wider the solved cube is across its opposite faces than
        designed (alignment-free, SCI2-6); "" without opposite faces."""
        pairs = self.visual.pairs()
        if not pairs:
            return ""
        lo, hi = min(p.change for p in pairs), max(p.change for p in pairs)
        if abs(hi - lo) < 1e-9:
            return _signed_length(hi, unit)
        if compact:  # "+423–511 µm"
            return f"{_signed_length(lo, unit).split(' ')[0]}–{_length(hi, unit)}"
        return f"{_signed_length(lo, unit)} to {_signed_length(hi, unit)}"

    def _squareness_sentence(self) -> str:
        """The largest alignment-free angle error between neighbouring faces (SCI2-6)."""
        angles = self.visual.squareness()
        if not angles:
            return ""
        a, b, dev = max(angles, key=lambda x: abs(x[2]))
        # Round 9 review: panel B labels each face's tilt; this sentence names that measure
        # (the largest tilt shown) and the neighbouring-face angle separately, so no number
        # is quoted under the wrong name.
        summary = self.visual.face_summary()
        tilt = (f"Faces tilt up to {summary['tilt_max']:.1f}° (B); neighbours"
                if summary else "Neighbours")
        return (f"{tilt} are up to {abs(dev):.1f}° off square "
                f"({es.face_label(a)}, {es.face_label(b)}); widths and angles do not depend on "
                "the alignment.")

    def _stage_shape(self) -> None:
        v, n = self.visual, self.narrative
        level = self.level()
        sizes, unit = v.correction_display()
        several_mm = n.value("several_directions_mean_displacement_mm")
        mean_text = "unavailable"
        if several_mm is not None and v.mm_per_unit is not None:
            mean_text = _length(float(several_mm), unit)
        summary = v.face_summary()
        chips = [Chip(mean_text, "mean shape change", tone="accent")]
        widths = self._width_range(unit)
        if summary is not None and widths:
            # Alignment-free (SCI2-6): the per-face offsets depend on the chosen turn + shift.
            chips.append(Chip(self._width_range(unit, compact=True), "cube wider than drawn"))
            if level >= 2:
                chips.append(Chip(_length(summary["inside_mean"], unit), "change inside faces"))
        elif summary is not None:
            chips.append(Chip(f"{_signed_length(summary['offset_min'], unit)} to "
                              f"{_signed_length(summary['offset_max'], unit)}",
                              "faces further out"))
            if level >= 2:
                chips.append(Chip(_length(summary["inside_mean"], unit), "change inside faces"))
        else:
            total = len(v.several) + len(v.one_direction) + len(v.unseen)
            chips.append(Chip(f"{len(v.several)} of {total}" if total else "unavailable",
                              "corners seen from several directions",
                              tone="good" if total and len(v.several) >= 0.8 * total else "warn"))
        self._set_chips(chips)
        self._morph_phase = self._morph_phase_of(self.morph.position())
        self.stage_view.set_caption(self._caption(self._shape_caption(self.morph.position())))
        self._numbers_shape()
        if not v.available or v.correction is None or sizes is None:
            reason = v.reason or "the display alignment needs at least three measured corners"
            self._unavailable_figure("Freeing the shape", f"Shape change not shown: {reason}.")
            return
        split = self._use_split()
        self.stage_view.set_controls(self.morph)
        self.morph.show()
        scene = split.scene
        scene.clear()
        Q = self._display_frame()
        t = es.stage_tokens(self._theme())
        printed = _to_display(Q, v.printed)
        several = np.array(v.several, dtype=int)
        one = np.array(v.one_direction, dtype=int)
        clim = _nice_clim(sizes[several] if several.size else sizes)
        self._morph_items = {"Q": Q, "several": several, "one": one, "clim": clim,
                             "factor": self.exaggeration,
                             "to_unit": v.lengths_in(unit) or 1.0}
        shown, moved = self._morph_positions(self.morph.position(), self.exaggeration)
        self._add_outline(scene, Q)
        scene.add_points(printed[np.r_[several, one]], name="printed", colour=t["muted"], size=4)
        if several.size:
            if self._theme().lower() != "dark":
                # Round 9 review: a dark halo under the coloured moves, so the yellow (largest)
                # ones keep their edge on the light themes (Dark needs none: yellow reads there)
                scene.add_rays(printed[several], shown[several], name="moves_halo",
                               colour=t["text"], width=4.6, opacity=0.5)
            scene.add_rays(printed[several], shown[several], name="moves",
                           scalars=moved[several], clim=clim, width=2.2, opacity=0.95)
            scene.add_points(shown[several], moved[several], name="refined", clim=clim,
                             scalar_bar=unit, size=9)
        if level >= 2 and one.size:
            scene.add_rays(printed[one], shown[one], name="moves_one", colour=t["muted"],
                           width=2.0, opacity=0.7)
            scene.add_points(shown[one], name="refined_one", colour=t["muted"], size=10)
        if v.unseen:
            scene.add_points(printed[np.array(v.unseen, dtype=int)], name="unseen",
                             colour=t["muted"], size=3)
        # ED3-3: F1-F6 on the cube, the face numbers the 2D panels use (the moved corners)
        scene.add_face_tags([(f.face, shown[np.asarray(f.points, dtype=int)]) for f in v.faces])
        self._view(scene)
        extras = []
        if level >= 2 and v.one_direction:
            extras.append("grey: one side only (×1)")
        split.set_titles("A  How each corner moved: printed → refined",
                         f"lines ×{self.exaggeration} longer · colour: distance ({unit}) "
                         "· squares keep size, faces further out"
                         + (" · " + " · ".join(extras) if extras else ""))
        has_faces = bool(v.faces) and v.plate is not None
        if level == 1 and not has_faces:
            split.set_side(None)
            return
        seps = v.ray_separation_deg
        factor = self.exaggeration

        def side(fig: Any, theme: str) -> None:
            tt = es.stage_tokens(theme)
            if has_faces:
                # L1: the slice; L2: slice over the per-face bars; L3: slice beside the bars
                # (each keeps a usable size) over what is left inside the faces.
                if level >= 3:
                    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0],
                                          width_ratios=[1.0, 1.05])
                    slots = (gs[0, 0], gs[0, 1], gs[1, :])
                else:
                    rows = [1.25] + ([1.0] if level >= 2 else [])
                    gs = fig.add_gridspec(len(rows), 1, height_ratios=rows)
                    slots = (gs[0], gs[1] if level >= 2 else None, None)
                ax = fig.add_subplot(slots[0])
                # Round 9: held labels tagged; the tag sits on its own line so the label stays
                # inside its panel (a one-line tag ran past the figure edge)
                tag = "" if self._usable() else "\n(not verified)"
                if not draw_face_slice(ax, v, self._rig_up(), factor, theme, unit, tag):
                    es.empty_panel(ax, "This target has no ring of faces to slice through.")
                # ED6-1 audit: on a held run the shape titles describe what was solved, not why
                # it is so (no explanation is backed by a held run).
                # ED7-1: held titles are short enough that "not verified" is never cut off
                held = not self._usable()
                es.panel_title(ax, "B", ("Solved cube, not verified" if held
                                         else "Why the cube is off" if level >= 3
                                         else "Why the cube is not its drawing"),
                               f"dashed: design · bars: as solved (×{factor})"
                               if level >= 3 else
                               f"a slice · dashed: design · bars: as solved (×{factor})")
                if level >= 2:
                    bx = fig.add_subplot(slots[1])
                    if v.pairs():
                        draw_face_pairs(bx, v.pairs(), theme, unit, v.faces, tag)
                        es.panel_title(bx, "C", ("Wider than drawn" if level >= 3
                                                 else "How much wider than drawn") if self._usable()
                                       else "Wider, not verified",
                                       "tick: each face's share")
                    else:
                        draw_face_bars(bx, v.faces, theme, unit, tag)
                        es.panel_title(bx, "C", ("Each face as a plate" if level >= 3
                                                 else "Each face as one rigid plate")
                                       if self._usable() else "Faces, not verified",
                                       "real sizes" if level >= 3
                                       else "fitted to its well-seen corners · real sizes")
                if level >= 3:
                    cx = fig.add_subplot(slots[2])
                    plate = v.plate
                    fitted = [i for f in v.faces for i in f.points if i in set(v.several)]
                    inside = (np.linalg.norm(v.refined[fitted] - plate[fitted], axis=1)
                              * (v.lengths_in(unit) or 1.0))
                    hi = float(np.percentile(inside, 99)) * 1.05 if inside.size else 1.0
                    counts, _e, _p = cx.hist(np.clip(inside, 0, hi),
                                             bins=np.linspace(0, hi, 26),
                                             color=tt["accent"], alpha=0.85, zorder=3)
                    es.integer_axis(cx, "y")
                    clipped = int(np.sum(inside > hi))
                    if clipped:  # SCI2-8: the last bin also holds the corners beyond it
                        cx.annotate(f"≥ {hi:.0f}" if unit == "µm" else f"≥ {hi:.2f}",
                                    (hi, counts[-1]), xytext=(0, 3), textcoords="offset points",
                                    ha="right", va="bottom", fontsize=8.5, color=tt["text"])
                    m = float(np.mean(inside))
                    cx.axvline(m, color=tt["text"], linestyle=(0, (4, 3)), linewidth=1.2,
                               zorder=4)
                    sub = f"dashes: mean {_length(m / (1000.0 if unit == 'µm' else 1.0), unit)}"
                    noise3, noise_mean = v.noise_3d()
                    if noise_mean is not None:
                        # SCI2-8: compare a 3D distance with the noise's expected 3D distance.
                        p = noise_mean * (1000.0 if unit == "µm" else 1.0)
                        cx.axvline(p, color=tt["good"], linestyle=(0, (1, 2)), linewidth=1.8,
                                   zorder=4)
                        sub += f" · dots: noise alone ≈ {_length(noise_mean, unit)}"
                    cx.set_xlabel(f"change left inside the face, corner by corner ({unit})")
                    cx.set_ylabel("corners")
                    es.panel_title(cx, "D", ("What is left inside each face" if self._usable()
                                             else "Inside faces, not verified"), sub)
                return
            rows = [1.0] + ([1.15] if level >= 3 else [])
            gs = fig.add_gridspec(len(rows), 1, height_ratios=rows)
            ax = fig.add_subplot(gs[0])
            draw_coloured_hist(ax, sizes[np.r_[several, one]] if several.size + one.size
                               else sizes, clim, theme)
            if several.size:
                m = float(np.mean(sizes[several]))
                ax.axvline(m, color=tt["text"], linestyle=(0, (4, 3)), linewidth=1.1)
                ax.annotate(f"mean {m:.3g} {unit}", (m, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(4, -4), textcoords="offset points", ha="left", va="top",
                            fontsize=8.5, color=tt["text"])
            ax.set_xlabel(f"size of the correction ({unit})")
            ax.set_ylabel("corners")
            es.panel_title(ax, "B", "How big the corrections are" + (
                               "" if self._usable() else " (not verified)"),
                           "real sizes, not exaggerated · bars coloured as in A")
            if level >= 3 and seps is not None:
                bx = fig.add_subplot(gs[1])
                for idx, colour, label in ((several, tt["accent"], "several directions"),
                                           (one, tt["muted"], "one direction")):
                    if idx.size:
                        bx.scatter(np.nan_to_num(seps[idx]), sizes[idx], s=14, color=colour,
                                   alpha=0.8, label=label, linewidths=0)
                es.limit_line(bx, THETA_MIN_DEG, values=seps, above_is_bad=False, axis="x")
                bx.set_xlabel("widest angle between the rays that saw the corner (°)")
                bx.set_ylabel(f"correction ({unit})")
                es.panel_title(bx, "C", ("More viewing directions pin a corner down"
                                         if self._usable() else
                                         "Viewing directions per corner (not verified)"),
                               f"left of {THETA_MIN_DEG:g}° (dashes) its depth is poorly "
                               "determined")
        split.set_side(side, (3, 2) if level < 3 or not has_faces else (5, 4))

    def _morph_positions(self, position: int, factor: int) -> tuple[np.ndarray, np.ndarray]:
        """(displayed corner positions, real size of each corner's move so far in the stage's
        unit) at morph ``position`` (0..20). Several-direction corners: first their face's
        plate move (positions 1..10), then the change inside the face (11..20), both ×
        ``factor``; corners on no face move linearly. One-direction corners move linearly ×1;
        unseen corners stay."""
        v = self.visual
        Q = self._morph_items.get("Q", np.eye(3))
        corr = np.nan_to_num(v.correction)
        p = float(position) / (MORPH_POSITIONS - 1)
        if v.plate is not None:
            plate = np.nan_to_num(v.plate - v.printed)
            on_face = np.all(np.isfinite(v.plate), axis=1)
            a = float(np.clip(position / MORPH_HALF, 0, 1))
            b = float(np.clip((position - MORPH_HALF) / MORPH_HALF, 0, 1))
            move = np.where(on_face[:, None], plate * a + (corr - plate) * b, corr * p)
        else:
            move = corr * p
        one = np.asarray(self._morph_items.get("one", v.one_direction), dtype=int)
        if one.size:
            move[one] = corr[one] * p
        scale = np.full(len(v.printed), float(factor))
        if one.size:
            scale[one] = 1.0
        shown = _to_display(Q, v.printed + move * scale[:, None])
        size = np.linalg.norm(move, axis=1) * float(self._morph_items.get("to_unit", 1.0))
        return shown, size

    def _morph_changed(self, position: int, factor: int) -> None:
        if self._showing != 1 or not self._morph_items:
            return
        if factor != self._morph_items.get("factor"):
            self._stage_shape()  # titles, chips, caption and the slice name the factor
            return
        phase = self._morph_phase_of(position)
        if phase != self._morph_phase:  # the caption follows the morph (ED-28)
            self._morph_phase = phase
            self.stage_view.set_caption(self._caption(self._shape_caption(position)))
        scene = self.split.scene
        v = self.visual
        printed = _to_display(self._morph_items["Q"], v.printed)
        shown, moved = self._morph_positions(position, factor)
        several, one = self._morph_items["several"], self._morph_items["one"]
        names = set(scene.names())
        frame: dict[str, Any] = {"points": {}, "rays": {}}
        if several.size and "refined" in names:
            frame["points"]["refined"] = {"xyz": shown[several], "scalars": moved[several]}
            frame["rays"]["moves"] = {"origins": printed[several], "ends": shown[several],
                                      "scalars": moved[several]}
        if one.size and "refined_one" in names:
            frame["points"]["refined_one"] = shown[one]
            frame["rays"]["moves_one"] = {"origins": printed[one], "ends": shown[one]}
        scene.apply_frame(frame)

    def _numbers_shape(self) -> None:
        n, v = self.narrative, self.visual
        level = self.level()
        unit = self._length_unit()
        rows = list(n.class_rows())
        sizes, _unit = v.correction_display()
        several = np.array(v.several, dtype=int)
        rows.append(("Mean shape change, several directions",
                     f"{fmt(n.value('several_directions_mean_displacement_mm'))} mm",
                     "turn + shift removed; size kept at the printed square size "
                     "(the scale the exported calibration uses)"))
        if sizes is not None and several.size:
            rows.append(("Largest shape change, several directions",
                         f"{fmt(float(np.nanmax(sizes[several])))} {_unit}", "same alignment"))
        rows.append(("D4.7 mean displacement" if level >= 2 else "Mean change, all seen corners",
                     f"{fmt(n.value('d47_mean_displacement_mm'))} mm",
                     "all seen corners, in the exported frame"))
        for face in v.faces:
            rows.append((f"Face {face.face + 1}",
                         f"{_signed_length(face.offset, unit)} out · tilt {face.tilt_deg:.2f}° · "
                         f"inside {_length(face.warp_mean, unit)} (max {_length(face.warp_max, unit)})",
                         f"rigid plate fitted to its {face.fitted} well-seen corners; slide "
                         f"{_length(face.shift, unit)} along the face"))
        for pair in v.pairs():
            a, b = (es.face_label(f, short=False) for f in pair.faces)
            rows.append((f"{a} and {b} (opposite)",
                         f"{_signed_length(pair.change, unit)} wider than the design's "
                         f"{_length(pair.design, 'mm')} · {pair.angle_deg:.2f}° off parallel",
                         "the two faces' outward moves added; unlike each face's own move, "
                         "this does not depend on the alignment"))
        angles = v.squareness()
        if angles:
            a, b, dev = max(angles, key=lambda x: abs(x[2]))
            rows.append(("Largest angle off square", f"{dev:+.2f}°",
                         f"{es.face_label(a, short=False)} and {es.face_label(b, short=False)}; "
                         "neighbouring solved plates against 90°, alignment-free"))
        summary = v.face_summary()
        if summary is not None:
            rows.append(("Change inside the faces", f"mean {_length(summary['inside_mean'], unit)}, "
                         f"max {_length(summary['inside_max'], unit)}",
                         "every well-seen corner, after its face's plate move"))
        if level >= 2:
            value, how = v.corner_precision() if level >= 3 else (None, "shown at L3")
            if level >= 3:
                rows.append(("Corner precision (formal, per coordinate)",
                             _length(value, unit) if value is not None else "not computed", how))
                noise3, noise_mean = v.noise_3d()
                if noise3 is not None:
                    rows.append(("Corner precision (3D)", f"{_length(noise3, unit)} (1σ)",
                                 f"√3 × the per-coordinate value; noise alone would move a "
                                 f"corner {_length(noise_mean, unit)} on average"))
        if level >= 3 and v.gauge_indices:
            axis = "xyz"[v.gauge_axis] if v.gauge_axis is not None else "one coordinate"
            ids = list(v.gauge_indices)
            rows.append(("Solver gauge (during the solve)",
                         f"corners {ids[0]} and {ids[1]} (x, y, z), corner {ids[2]} ({axis})"
                         if len(ids) == 3 else ", ".join(map(str, ids)),
                         "their coordinates are set to the printed values and removed from the "
                         "unknowns, so the shape has one specific form (stored indices, D4.2); "
                         "the result is re-aligned afterwards, so these corners move like any "
                         "other"))
        rows.append(("Exaggeration", f"×{self.exaggeration}",
                     "several-direction corrections only; display only"))
        rows.append(("What the shape shows", "", n.shape_copy()))
        if level >= 2 and n.shape_detail():
            rows.append(("Behind the mean", "", n.shape_detail()))
        if level >= 3:
            row = n.refit_scale_row()
            if row is not None:
                rows.append(row)
        self._numbers(["Item", "Value", "Evidence"], rows)

    # -- stage 3 -------------------------------------------------------------------------

    def _stage_solver(self) -> None:
        v, pb = self.visual, self.playback
        level = self.level()
        steps = len(pb.states) - 1 if pb.playable else 0
        last = pb.states[-1].nit if pb.playable else None
        chips = [Chip(f"{steps} of {last}" if pb.playable else pb.status, "steps recorded",
                      tone=None if pb.playable else "warn")]
        if pb.playable:
            # ED5-6 / SCI4-2: a held run's change is shown neutrally (no green or red verdict).
            chips.append(Chip(es.format_change(pb.states[0].mean_px, pb.states[-1].mean_px),
                              "mean error", tone=("good" if pb.states[-1].mean_px
                                                  < pb.states[0].mean_px else "bad")
                              if self._usable() else None))
        self._set_chips(chips)
        self._numbers_solver()
        if not pb.playable:
            self._stage_solver_static(level)
            return
        plates = self._solver_plates_available()
        factor = self._solver_factor()
        # SCI4-2: the RMS claim is about this run's own steps, never a general promise.
        rms = [float(np.sqrt(np.mean(np.square(s.norms)))) if len(s.norms) else math.nan
               for s in pb.states]
        finite_rms = bool(np.isfinite(rms[0]) and np.isfinite(rms[-1]))
        loss_name = self.narrative.phase4_loss()[0]
        if finite_rms and loss_name != "linear":
            # SCI5-5: a robust loss is not squared error, so the RMS need not fall; say so and
            # do not blame the solver for not minimising what it does not minimise.
            fell = "fell" if rms[-1] < rms[0] else "did not fall"
            rms_line = (f"The solver minimises a robust loss ({loss_name}), not squared errors, so "
                        f"the dashed RMS need not fall; here it {fell} ({rms[0]:.2f} → "
                        f"{rms[-1]:.2f} px). The strip shows which parameters moved.")
        elif finite_rms and rms[-1] < rms[0]:
            rms_line = (f"The solver minimises squared errors, so the dashed RMS falls here, "
                        f"{rms[0]:.2f} → {rms[-1]:.2f} px; the strip shows which parameters moved.")
        elif finite_rms:
            rms_line = (f"On this run the dashed RMS did not fall ({rms[0]:.2f} → {rms[-1]:.2f} px), "
                        "although the solver minimises squared errors; see 'Can we trust it?'.")
        else:
            rms_line = ("The dashed line is the RMS at each step; the strip shows which "
                        "parameters moved.")
        caption = [
            "Press Play: at every accepted step the solver adjusts the cameras, the lenses, the "
            "target's position in each photo and its corners together.",
            ("Each face rides as one plate (stage 2), its move drawn ×"
             f"{factor}; dots are coloured by their error at that step." if plates else
             f"Corners are drawn at their solved positions (moves ×{factor}), coloured by their "
             "error at that step."),
            (f"Most of the {'drop' if self._usable() else 'change'} comes in the first steps; grey "
             "trails show where each face travelled." if plates else
             f"Most of the {'drop' if self._usable() else 'change'} comes in the first steps."),
            rms_line,
        ]
        self.stage_view.set_caption(self._caption(caption))
        split = self._use_split()
        scene = split.scene
        scene.clear()
        Q = self._display_frame()
        t = es.stage_tokens(self._theme())
        idx = self._solver_indices()
        scale = es.error_scale(self._theme())
        frames = [self._solver_frame(state, Q, level) for state in pb.states]
        first = frames[0]
        self._add_cameras(scene, Q, pb.states[0].cameras)
        self._add_outline(scene, Q)
        if plates:
            for name, corners in first.get("boards", {}).items():
                face = int(name.removeprefix("plate"))
                scene.add_board(corners, colour=es.face_colour(face, self._theme()),
                                name=name, opacity=0.15, point_size=1.0)
        if level >= 3:  # thin lines back to the printed design (UX2-4: L3 only, thinned)
            printed = _to_display(Q, v.printed)
            scene.add_rays(printed[idx], first["points"]["shape"]["xyz"], name="moves",
                           colour=t["muted"], width=0.8, opacity=0.35)
        scene.add_points(first["points"]["shape"]["xyz"], first["points"]["shape"]["scalars"],
                         name="shape", scalar_bar="px", size=9, **scale.scene_kwargs())
        self._view(scene)
        self._face_trails = {}
        if plates and level >= 2:
            self._add_face_trails(scene, 0.0)
        split.set_titles("A  The solver at work",
                         ("faces as plates, moves ×" if plates else "corner moves ×")
                         + f"{factor} · colour: error at this step (px, log)"
                         + (" · trails: face paths" if plates and level >= 2 else ""))
        weights = pace_weights([s.mean_px for s in pb.states])
        if self.player is None:
            self.player = ScenePlayer(scene, frames, step_numbers=[s.nit for s in pb.states[1:]],
                                      total_steps=last, step_weights=weights or None,
                                      parent=self)
            self.player.frameShown.connect(self._solver_frame_shown)
            self.player.stepReached.connect(self._solver_step_reached)
            self._add_solver_exaggeration_menu()
        else:
            self.player.scene = scene
            self.player.frames = [dict(f) for f in frames]
        self._sync_solver_exaggeration_menu()
        self.stage_view.set_controls(self.player)
        self._solver_index = int(round(self._play_position))
        split.set_side(lambda fig, theme: self._draw_solver_side(fig, theme, level), (3, 2))
        self.player.go_to(int(round(self._play_position)), animate=False)

    # The solver's exaggeration is stage 2's (one ⚙ "Exaggerate" choice on both playback
    # bars, UX2-21); without face plates it is capped so that no corner is drawn more than
    # SOLVER_CAP of the target size from its design (UX2-4).
    SOLVER_CAP = 0.03

    def _solver_plates_available(self) -> bool:
        return bool(self.visual.faces) and self.visual.plate is not None

    def _solver_factor(self) -> int:
        factor = int(self.exaggeration)
        if self._solver_plates_available():
            return factor
        v = self.visual
        largest = 0.0
        for state in self.playback.states:
            shape = state.display_shape if state.display_shape is not None else state.shape
            moves = np.linalg.norm(np.asarray(shape, dtype=float) - v.printed, axis=1)
            moves = moves[np.isfinite(moves)]
            largest = max(largest, float(moves.max()) if moves.size else 0.0)
        if largest <= 0:
            return factor
        cap = self.SOLVER_CAP * v.target_size / largest
        allowed = [f for f in self._exaggeration_choices() if f <= max(cap, 1.0)]
        return int(min(factor, max(allowed or [1])))

    def _add_solver_exaggeration_menu(self) -> None:
        controls = getattr(self.player, "controls", None)
        if controls is None or not hasattr(controls, "add_menu_choice"):
            self._solver_menu = None
            return
        factors = self.morph.factors
        self._solver_menu = controls.add_menu_choice(
            "Exaggerate", [f"×{f}" for f in factors],
            factors.index(self.exaggeration) if self.exaggeration in factors else 0,
            self._solver_factor_picked)

    def _sync_solver_exaggeration_menu(self) -> None:
        menu = getattr(self, "_solver_menu", None)
        if menu is None:
            return
        for k, action in enumerate(menu.actions()):
            action.setChecked(self.morph.factors[k] == self.exaggeration)

    def _solver_factor_picked(self, index: int) -> None:
        factor = self.morph.factors[int(index)]
        if factor == self.exaggeration:
            return
        if self.player is not None:
            self._play_position = self.player.position
        self.morph.set_exaggeration(factor)
        if self._showing == 2:
            self._stage_solver()

    def _solver_indices(self) -> np.ndarray:
        v = self.visual
        return np.array(list(v.several) + list(v.one_direction), dtype=int)

    def _state_plates(self, state: Any) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Each face's rigid plate (R, t) for one solver state's aligned shape."""
        cache = self.__dict__.setdefault("_plate_cache", {})
        key = (id(self.visual), int(state.nit))
        if key not in cache:
            v = self.visual
            from pyCamSet.workflow.education_visual_p4 import decompose_faces
            display = state.display_shape if state.display_shape is not None else state.shape
            faces = [np.asarray(f.points, dtype=int) for f in v.faces]
            moves, _plate = decompose_faces(v.printed, np.asarray(display, dtype=float),
                                            v.several, faces)
            by_points = {tuple(f.points): f.face for f in v.faces}
            cache[key] = {by_points.get(m.points, m.face): (m.R, m.t) for m in moves}
        return cache[key]

    def _solver_shape(self, state: Any) -> np.ndarray:
        """Displayed corner positions (target frame, all points) for one state: each face's
        corners on its plate, the plate's move × the factor (several-direction cube); else
        each corner's own move × the (capped) factor, one-direction corners ×1."""
        v = self.visual
        factor = float(self._solver_factor())
        display = state.display_shape if state.display_shape is not None else state.shape
        corr = np.nan_to_num(np.asarray(display, dtype=float) - v.printed)
        if self._solver_plates_available():
            shown = v.printed + corr  # corners on no fitted face: their real position
            plates = self._state_plates(state)
            for face in v.faces:
                if face.face not in plates:
                    continue
                R, t = plates[face.face]
                idx = np.asarray(face.points, dtype=int)
                moved = v.printed[idx] @ np.asarray(R).T + np.asarray(t)
                shown[idx] = v.printed[idx] + factor * (moved - v.printed[idx])
            return shown
        scale = np.full(len(v.printed), factor)
        scale[list(v.one_direction)] = 1.0
        return v.printed + corr * scale[:, None]

    def _plate_corners(self, face: Any, shown: np.ndarray) -> np.ndarray:
        """The 4 corners of a face's printed area, carried by the displayed plate."""
        v = self.visual
        idx = np.asarray(face.points, dtype=int)
        pts = v.printed[idx]
        centre = pts.mean(axis=0)
        _u, _s, vt = np.linalg.svd(pts - centre)
        a, b = vt[0], vt[1]
        ca, cb = (pts - centre) @ a, (pts - centre) @ b
        corners_idx = [idx[np.argmin(ca + cb)], idx[np.argmax(ca - cb)],
                       idx[np.argmax(ca + cb)], idx[np.argmax(cb - ca)]]
        return shown[corners_idx]

    def _solver_frame(self, state: Any, Q: np.ndarray, level: int) -> dict[str, Any]:
        """Scene frame for one solver state: cameras, corners (on their face plates, or
        capped per-corner moves) coloured by their error at that state, the plates and, at
        L3, the thin lines back to the printed design."""
        v = self.visual
        idx = self._solver_indices()
        shape = self._solver_shape(state)
        shown = _to_display(Q, shape)[idx]
        errors = (np.nan_to_num(state.point_error[idx]) if len(state.point_error)
                  else np.zeros(len(idx)))
        frame: dict[str, Any] = {
            "cameras": {f"cam{c.index}": {"R": _camera_display(Q, c)[0],
                                          "t": _camera_display(Q, c)[1]} for c in state.cameras},
            "points": {"shape": {"xyz": shown, "scalars": errors}},
        }
        if self._solver_plates_available():
            frame["boards"] = {f"plate{f.face}": _to_display(Q, self._plate_corners(f, shape))
                               for f in v.faces}
        if level >= 3:
            frame["rays"] = {"moves": {"origins": _to_display(Q, v.printed)[idx], "ends": shown}}
        return frame

    def _face_centre_path(self, face: Any, position: float) -> np.ndarray:
        """Displayed centre of ``face`` at every state up to ``position`` (interpolated)."""
        Q = self._display_frame()
        states = self.playback.states
        cache = self.__dict__.setdefault("_centre_cache", {})
        key = (id(self.visual), face.face, self._solver_factor())
        if key not in cache:
            idx = np.asarray(face.points, dtype=int)
            cache[key] = np.array([_to_display(Q, self._solver_shape(s)[idx].mean(axis=0))[0]
                                   for s in states])
        centres = cache[key]
        lo = int(math.floor(max(0.0, min(position, len(states) - 1))))
        path = list(centres[:lo + 1])
        frac = position - lo
        if frac > 0 and lo + 1 < len(states):
            path.append(centres[lo] + frac * (centres[lo + 1] - centres[lo]))
        return np.asarray(path)

    def _add_face_trails(self, scene: Any, position: float) -> None:
        """One trail per face plate (UX2-6 primitive): where its centre travelled so far."""
        colour = es.stage_tokens(self._theme())["muted"]
        for face in self.visual.faces:
            try:
                name = scene.add_trail(self._face_centre_path(face, len(self.playback.states)),
                                       name=f"facetrail{face.face}", colour=colour,
                                       width=2.4, min_fraction=0.004)
                scene.update_trail(name, self._face_centre_path(face, position))
            except Exception:  # noqa: BLE001 - a scene without trails still plays
                return
            self._face_trails[face.face] = name

    def _solver_frame_shown(self, position: float, _frame: Any) -> None:
        # The player renders the 3D scene once after this handler (render=False here).
        if self._showing == 2 and getattr(self, "_face_trails", None):
            scene = self.split.scene
            for face in self.visual.faces:
                name = self._face_trails.get(face.face)
                if name is not None:
                    try:
                        scene.update_trail(name, self._face_centre_path(face, position),
                                           render=False)
                    except Exception:  # noqa: BLE001
                        pass
        marker = getattr(self, "_cost_marker", None)
        pb = self.playback
        if marker is None or not pb.playable:
            return
        lo = int(math.floor(position))
        hi = min(lo + 1, len(pb.states) - 1)
        s = position - lo
        a, b = pb.states[lo], pb.states[hi]
        x = a.nit + s * (b.nit - a.nit)
        y = math.exp((1 - s) * math.log(a.mean_px) + s * math.log(b.mean_px)) \
            if a.mean_px > 0 and b.mean_px > 0 else a.mean_px
        marker.set_data([x], [y])
        self._request_side_draw(position)

    def _request_side_draw(self, position: float) -> None:
        """Redraw the 2D side panel at most every REDRAW_INTERVAL_MS while a solver frame
        plays. A whole step (and so the last frame) is drawn at once; a half-way frame that
        comes sooner is deferred to a single-shot timer, never dropped. The marker is always
        updated, so the deferred draw shows the latest position."""
        if getattr(self, "_side_clock", None) is None:
            self._side_clock = QElapsedTimer()
            self._side_clock.start()
            self._side_timer = QTimer(self)
            self._side_timer.setSingleShot(True)
            self._side_timer.timeout.connect(self._flush_side_draw)
        whole = abs(position - round(position)) < 1e-9
        wait = perf.REDRAW_INTERVAL_MS - self._side_clock.elapsed()
        if not whole and wait > 0:
            if not self._side_timer.isActive():
                self._side_timer.start(int(wait) + 1)
            return
        self._side_timer.stop()
        self._side_clock.restart()
        self.split.canvas.draw_idle()

    def _flush_side_draw(self) -> None:
        split = getattr(self, "split", None)
        canvas = getattr(split, "canvas", None)
        if canvas is not None and self._showing == 2:
            self._side_clock.restart()
            canvas.draw_idle()

    def _solver_step_reached(self, index: int) -> None:
        if self._showing != 2:
            return
        if index != getattr(self, "_solver_index", None):
            self._solver_index = index
            self.split.redraw_side()

    def _draw_solver_side(self, fig: Any, theme: str, level: int) -> None:
        pb = self.playback
        t = es.stage_tokens(theme)
        k = max(0, min(getattr(self, "_solver_index", 0), len(pb.states) - 1))
        rows = [1.0, 1.0] + ([0.8] if level >= 3 else [])
        gs = fig.add_gridspec(len(rows), 1, height_ratios=rows)
        ax = fig.add_subplot(gs[0])
        nits = [s.nit for s in pb.states]
        means = [s.mean_px for s in pb.states]
        ax.plot(nits, means, color=t["muted"], linewidth=1.4, marker="o", markersize=3.5, zorder=2)
        ax.plot(nits[:k + 1], means[:k + 1], color=t["accent"], linewidth=2.2, zorder=3)
        if level >= 3:
            rms = [float(np.sqrt(np.mean(np.square(s.norms)))) if len(s.norms) else math.nan
                   for s in pb.states]
            ax.plot(nits, rms, color=t["text"], linewidth=1.1, linestyle=(0, (4, 3)), zorder=2,
                    label="RMS")
            # Round 9 review: the RMS key sits lower left, clear of its dashed line.
            ax.legend(loc="lower left", fontsize=8.5)
        (self._cost_marker,) = ax.plot([nits[k]], [means[k]], marker="o", markersize=10,
                                       color=t["accent"], markeredgecolor=t["text"],
                                       markeredgewidth=1.0, zorder=4, linestyle="")
        scale = es.error_scale(theme)
        scale.log_axis(ax, "y")
        every = [m for m in means if m > 0] + ([r for r in rms if np.isfinite(r) and r > 0]
                                              if level >= 3 else [])
        if every:
            ax.set_ylim(min(every) * 0.7, max(every) * 1.5)
        ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("accepted solver step")
        ax.set_ylabel(es.error_label("mean", short=True))
        label = "start" if k == 0 else f"step {nits[k]}"
        held = "" if self._usable() else " (not verified)"  # ED7-2: held numbers are tagged
        es.panel_title(ax, "B", "Error at each step", f"{label} · mean {means[k]:.2f} px{held}"
                       + (" · dashes: RMS" if level >= 3 else ""))
        bx = fig.add_subplot(gs[1])
        every = [x for s in pb.states for x in s.per_camera.values()]
        draw_camera_log_bars(bx, pb.states[k].per_camera, pb.states[0].per_camera, every,
                             theme)
        es.panel_title(bx, "C", "Each camera's error at this step",
                       "log scale · grey ticks: at the start" + held)  # ED7-2: tagged when held
        if level >= 3 and pb.block_share is not None:
            cx = fig.add_subplot(gs[2])
            share = pb.block_share[1:].T
            # Shares are amounts, not errors: the amount map, with its own bar (UX3-3).
            scale = es.amount_scale(share, theme, unit="share")
            image = cx.imshow(share, aspect="auto", extent=(nits[1] - 0.5, nits[-1] + 0.5, 3.5, -0.5),
                              **scale.imshow_kwargs())
            scale.colourbar(fig, cx, image)
            sizes = pb.block_sizes or (0, 0, 0, 0)
            names = ["lenses", "camera turns", "target positions", "target corners"]
            cx.set_yticks(range(4))
            cx.set_yticklabels([f"{nm} ({sz})" for nm, sz in zip(names, sizes)], fontsize=8.5)
            if k >= 1:
                cx.axvline(nits[k], **es.cursor_style(cx))
            cx.set_xlabel("accepted solver step")
            for spine in cx.spines.values():
                spine.set_visible(False)
            es.panel_title(cx, "D", "Which parameters moved",
                           "each group's share of its change, per step")

    def _stage_solver_static(self, level: int) -> None:
        """No playable steps: show why, with the recorded costs when there are any."""
        v, n, pb = self.visual, self.narrative, self.playback
        copy = (n.failed_playback_copy() if self.run_context.status == "failed"
                else n.playback_copy(False))
        self.stage_view.set_caption(self._caption([copy.split(". ")[0].rstrip(".") + ".",
                                                   pb.reason[:1].upper() + pb.reason[1:] + "."
                                                   if pb.reason else ""]))
        costs = pb.costs

        def draw(fig: Any, theme: str) -> None:
            t = es.stage_tokens(theme)
            ax = fig.add_subplot()
            if costs:
                ax.plot([c[0] for c in costs], [c[1] for c in costs], color=t["accent"],
                        marker="o")
                ax.set_yscale("log")
                ax.set_xlabel("solver step (recorded)")
                # SCI6-1: a robust loss's cost is not a sum of squares, so the axis says so.
                ax.set_ylabel(cost_axis_label(self.narrative.phase4_loss()[0]))
                ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
                es.panel_title(ax, None, "Recorded cost only", pb.reason)
            else:
                start = v.start.mean_px if v.start is not None else None
                final = v.final.mean_px if v.final is not None else None
                if start is not None and final is not None:
                    ax.bar([0, 1], [start, final], color=[t["muted"], t["accent"]], width=0.55)
                    ax.set_xticks([0, 1], ["start", "result"])
                    ax.set_ylabel(es.error_label("mean", short=True))
                    for x, y in ((0, start), (1, final)):
                        ax.annotate(f"{y:.3g} px", (x, y), xytext=(0, 3),
                                    textcoords="offset points", ha="center", va="bottom")
                    es.panel_title(ax, None, "Start and result only",
                                   "the steps in between were not recorded; no path is invented")
                else:
                    es.empty_panel(ax, f"No solver steps to show: {pb.reason}.")
                    es.panel_title(ax, None, "Watching the solver", None)
        self.stage_view.set_controls(None)
        self.stage_view.draw_figure(draw)
        del level

    def _numbers_solver(self) -> None:
        n, pb = self.narrative, self.playback
        if self.run_context.status == "failed":
            rows = [("Phase 4 result", "failed", "selected run record"), *n.failed_trace_rows(),
                    ("About these steps", "", n.failed_playback_copy())]
            self._numbers(["Phase 4 solver step", "Recorded cost", "Evidence"], rows)
            return
        if not pb.playable:
            # The derivation's own recorded frames (validated against the saved run), as the
            # earlier text-first view listed them; no path is drawn between them.
            frames = tuple(getattr(self._derivation, "frames", ()) or ())
            rows3 = (step_rows(frames, self._largest_rig_change) if frames
                     else [("Playback", pb.status, pb.reason)])
            rows3.append(("About these steps", "", n.playback_copy(False)))
            self._numbers(["Solver step", cost_column_title(self.narrative.phase4_loss()[0]),
                           "Largest rig change"], rows3)
            return
        rows: list[tuple[str, ...]] = []
        prev = None
        for state in pb.states:
            label = "Starting point (not an accepted step)" if state.nit == 0 else f"Step {state.nit}"
            turn = "—"
            if prev is not None:
                turn = self._largest_camera_change(prev, state)
            rows.append((label, fmt(state.cost), f"{state.mean_px:.4g}", turn))
            prev = state
        if pb.missing_steps:
            rows.append(("Thinning", "steps not kept", ", ".join(map(str, pb.missing_steps)), ""))
        rows.append(("About these steps", "", n.playback_copy(pb.playable), ""))
        self._numbers(["Solver step", cost_column_title(self.narrative.phase4_loss()[0]),
                       "Mean error (px)", "Largest camera change"], rows)

    def _largest_rig_change(self, left: Any, right: Any) -> str:
        """Camera-centre movement between two derive_phase4 frames (scale-qualified like the
        rig table); "unavailable" unless both frames carry the same cameras."""
        try:
            left_names = list(left.rig_cameras.get_names())
            if left_names != list(right.rig_cameras.get_names()):
                return "unavailable"
            distances = [np.linalg.norm(np.asarray(left.rig_cameras[name].position, dtype=float)
                                        - np.asarray(right.rig_cameras[name].position, dtype=float))
                         for name in left_names]
            unit, scale = self.narrative.display_unit()
            qualified = (" (scale-qualified)"
                         if getattr(self._derivation, "metric_qualifier_required", False) else "")
            return f"{fmt(max(distances, default=0.0) * (scale or 1.0))} {unit}{qualified}"
        except Exception:
            return "unavailable"

    def _largest_camera_change(self, left: Any, right: Any) -> str:
        """Camera-centre movement (pinhole, scale-qualified like the rig table) or the largest
        turn (telecentric: there is no position)."""
        turns, moves = [], []
        for a, b in zip(left.cameras, right.cameras):
            rel = np.asarray(b.R) @ np.asarray(a.R).T
            turns.append(math.degrees(math.acos(float(np.clip((np.trace(rel) - 1) / 2, -1, 1)))))
            if not a.telecentric:
                moves.append(float(np.linalg.norm(a.centre - b.centre)))
        if moves:
            unit, scale = self.narrative.display_unit()
            qualified = (" (scale-qualified)"
                         if getattr(self._derivation, "metric_qualifier_required", False) else "")
            return f"{fmt(max(moves) * (scale or 1.0))} {unit}{qualified}"
        return f"{fmt(max(turns, default=0.0))}° turn"

    # -- stage 4 -------------------------------------------------------------------------

    def _stage_trust(self) -> None:
        v = self.visual
        level = self.level()
        status = self.run_context.status
        verdict = ("Usable", "good") if status == "complete" and not v.blocking else \
            ("Failed", "bad") if status == "failed" else ("Held back", "bad")
        nullity = v.nullity or {}
        # UX-37: the cards carry the checks; the chips only add what the cards do not say.
        chips = [Chip(verdict[0], "quality gate", tone=verdict[1])]
        shape = nullity.get("jacobian_shape") or ()
        unknowns = nullity.get("free_parameters")
        if len(shape) == 2 and unknowns:
            # UX5-4: plain words for the ratio, not "per unknown".
            chips.append(Chip(f"{int(shape[0]) / int(unknowns):.0f} numbers",
                              "for each unknown"))
        self._set_chips(chips)
        unit = self._length_unit()
        self.stage_view.set_caption(self._caption(self._trust_caption(unit)))
        # The look-alike panel and the rank check use two terms the caption does not name, so the
        # Terms button lists them too (L2+). The automatic terms of the caption and chips stay.
        found = es.glossary_terms_in(" ".join(
            [self.stage_view.caption.text()] + [c.label for c in self.stage_view.chips.chips()]))
        extra = (["gauge freedom"] if not v.blocking else []) + (["null space"] if nullity else [])
        self.stage_view.set_glossary(list(dict.fromkeys([*found, *extra])))
        self._numbers_trust()
        cov = v.coverage
        Q = self._display_frame()

        def draw(fig: Any, theme: str) -> None:
            t = es.stage_tokens(theme)

            def held_back(ax: Any, letter: str) -> None:
                # UX3-4: the gate's internal flags in plain words, each on a red-edged card with
                # what to do; the photos left out are counted, not listed.
                from matplotlib.patches import FancyBboxPatch
                ax.set_axis_off()
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                reasons = list(v.blocking[:3]) or ["the quality gate did not pass"]
                step = 0.96 / len(reasons)
                for i, reason in enumerate(reasons):
                    what, todo = plain_blocking(reason, len(v.excluded))
                    top = 0.98 - i * step
                    card = FancyBboxPatch((0.01, top - step + 0.03), 0.98, step - 0.06,
                                          boxstyle="round,pad=0,rounding_size=0.04",
                                          facecolor=theme_tokens_for(theme)["surface_alt"],
                                          edgecolor=t["bad"],
                                          linewidth=1.4, transform=ax.transAxes, clip_on=False)
                    ax.add_patch(card)
                    # UX4-1 / UX5-2: the text is centred in its card, so a short reason
                    # leaves no empty strip inside the red edge.
                    es.card_text(ax, 0.05, top - step / 2, f"✕  {what}  To do: {todo}",
                                 card=card, transform=ax.transAxes, ha="left", va="center",
                                 fontsize=9.5, color=t["text"], max_lines=4)
                es.panel_title(ax, letter, "Why it is held back",
                               "what the gate found, and what to do next")

            def coverage(ax: Any, letter: str) -> None:
                if cov is not None:
                    draw_coverage(ax, cov, v.excluded, theme,
                                  badge_size=6.5 if level >= 3 else 9.0)
                    if level >= 3:  # the photo axis is named in the subtitle instead
                        ax.set_xlabel("")
                    es.panel_title(ax, letter, "Who saw which photo",
                                   ("x: photo · " if level >= 3 else "")
                                   + "one row per camera · darker = more corners"
                                   + (" · red = left out" if v.excluded else ""))
                else:
                    es.empty_panel(ax, "Which camera saw which photo was not recorded.")
                    es.panel_title(ax, letter, "Who saw which photo", None)

            def noise_panel(ax: Any, letter: str) -> None:
                # ED9-1: the bottom bar is the cube's own size error; say so. At L1 and L3 it is
                # a line under the axis (L3 broken after the semicolon). At L2 the panel sits
                # above panel E with no room for a second line, so the note goes in the subtitle.
                if not self._usable():
                    note = "bottom bar: the solved cube came out wider than drawn (not verified)"
                elif level >= 3:
                    note = ("bottom bar: the printed cube came out wider than drawn;\n"
                            "Phase 4 learns this")
                else:
                    note = ("bottom bar: the printed cube came out wider than drawn; "
                            "Phase 4 learns this")
                if level == 2:
                    draw_noise_check(ax, v, theme, unit)
                    sub = (f"{unit}, log · dashes: noise alone · bottom: solved cube wider "
                           "(not verified)" if not self._usable() else
                           f"{unit}, log · dashes: noise alone · bottom: printed cube wider; "
                           "Phase 4 learns it")
                else:
                    draw_noise_check(ax, v, theme, unit, note)
                    sub = f"{unit}, log · dashes: what noise alone would give"
                es.panel_title(ax, letter, "Bigger than its noise?", sub)

            def unknowns_panel(ax: Any, letter: str, define: bool = False, short: bool = False) -> None:
                draw_unknowns(ax, v, theme, define=define, short=short)
                es.panel_title(ax, letter, "Far more measurements than unknowns", None)

            def gauge_or_held(spec: Any, letter: str) -> None:
                if v.blocking:
                    held_back(fig.add_subplot(spec), letter)
                else:
                    draw_gauge_strip(fig, spec, v, Q, theme, letter=letter,
                                     compact=level >= 3)

            if v.blocking and cov is not None:
                # Held back (UX3-4): the photos are the picture and the reason is a full-width
                # card, at every level (the side-by-side layout leaves too little room for both).
                # UX4-1: the held-back card is as tall as its reasons need (one reason ~2 lines).
                held_h = 0.45 + 0.25 * (min(len(v.blocking), 3) - 1)
                gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.8 if level == 1 else 2.1,
                                                           held_h, 0.5])
                draw_gate_cards(fig, gs[0, 0], v.checks, theme, compact=level >= 3)
                coverage(fig.add_subplot(gs[1, 0]), "B")
                held_back(fig.add_subplot(gs[2, 0]), "C")
                unknowns_panel(fig.add_subplot(gs[3, 0]), "D", define=True)
            elif level == 1:
                gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 2.0, 0.5],
                                      width_ratios=[1.1, 1.0])
                draw_gate_cards(fig, gs[0, :], v.checks, theme)
                noise_panel(fig.add_subplot(gs[1, 0]), "B")
                gauge_or_held(gs[1, 1], "C")
                unknowns_panel(fig.add_subplot(gs[2, :]), "D", define=True)
            elif level == 2:
                gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.7, 1.45],
                                      width_ratios=[1.3, 1.0])
                draw_gate_cards(fig, gs[0, :], v.checks, theme)
                coverage(fig.add_subplot(gs[1, 0]), "B")
                noise_panel(fig.add_subplot(gs[1, 1]), "C")
                low = gs[2, 0].subgridspec(2, 1, height_ratios=[1.0, 1.0])
                unknowns_panel(fig.add_subplot(low[0]), "D", short=True)  # a narrower panel
                gauge_or_held(gs[2, 1], "E")
            else:
                # the photo grid needs ~20 px a camera row for its badges (8 rows at L3)
                gs = fig.add_gridspec(4, 3, height_ratios=[0.95, 1.85, 1.25, 0.42],
                                      width_ratios=[1.1, 1.0, 1.1])
                draw_gate_cards(fig, gs[0, :], v.checks, theme, compact=True)
                coverage(fig.add_subplot(gs[1, :2]), "B")
                noise_panel(fig.add_subplot(gs[1, 2]), "C")
                gauge_or_held(gs[2, :2], "D")
                draw_rank_check(fig.add_subplot(gs[2, 2]), nullity, theme, "E")
                ux = fig.add_subplot(gs[3, :])
                draw_unknowns(ux, v, theme, equation=True)
                es.panel_title(ux, "F", "Far more measurements than unknowns", None)
        self.stage_view.set_controls(None)
        self.stage_view.draw_figure(draw)

    def _trust_caption(self, unit: str) -> list[str]:
        """The checks; why they are real evidence (ED2-6: the change against its noise, not
        "the error went down"); how much freedom alone could explain (SCI2-9); and at L3 what
        the rank check does and does not say (SCI2-3)."""
        v = self.visual
        nullity = v.nullity or {}
        first = ("Five checks run before a result is handed on: green passed, amber a notice, "
                 "red held back.")
        rows = noise_rows(v, unit)
        noise = rows[0][1] if rows else None
        changes = [r for r in rows if r[3] == "change"]
        if noise and changes:
            k = 1000.0 if unit == "µm" else 1.0
            letter = "B" if self.level() == 1 else "C"
            # Round 9 review: 4.7 is the MEAN change inside the faces (48 µm over about 10 µm of
            # noise), so it is stated as an average, never as a minimum (round down: SCI4-3).
            means = [r for r in changes if r[0] in ("change inside the faces", "mean corner change")]
            if means:
                ratio = math.floor(means[0][1] / noise * 10) / 10
                if means[0][0] == "change inside the faces":
                    claim = f"the shape changes, on average, {ratio:.1f}× what noise alone gives"
                else:
                    claim = f"the mean corner change is {ratio:.1f}× what noise alone gives"
                second = (f"The real test is {letter}: {claim} (≈ {_length(noise / k, unit)}), "
                          "so it is not detection noise.")
            else:
                low = min(r[1] for r in changes)
                bound = math.floor(low / noise * 10) / 10
                second = (f"The real test is {letter}: every cube width change is at least "
                          f"{bound:.1f}× what noise alone gives (≈ {_length(noise / k, unit)}), "
                          "so it is not detection noise.")
        else:
            second = ("A lower error alone proves little, since extra freedom lowers the solver's "
                      "objective; the shape needs far more measurements than unknowns.")
        drop = v.freedom_drop()
        free = nullity.get("free_parameters")
        rank = nullity.get("rank")
        p_eff = int(rank) if rank else (int(free) if free else None)
        rms0, rms1 = _rms_of(v.start_errors), _rms_of(v.final_errors)
        if drop is not None and p_eff and rms0 and rms1 is not None:
            # SCI3-6: the RMS drop is compared with the RMS statement (like with like).
            # SCI4-2: the sentence says what this run did; a rise is never called a fall.
            # SCI4-5: the formula uses the rank, so the caption counts the rank too.
            # Round 9 review: the same unknowns phrase as the panel (1,466 unknowns, 3 not pinned).
            if not free or int(free) == p_eff:
                count = f"{p_eff:,} effective unknowns"
            else:
                count = (f"{int(free):,} unknowns, of which {int(free) - p_eff:,} are not pinned "
                         f"down (rank {p_eff:,})")
            if rms1 < rms0 and self._usable():
                third = (f"Fitting noise with {count} could lower the RMS error by "
                         f"only about {drop:.0%}; it fell {1 - rms1 / rms0:.0%} "
                         f"({rms0:.2f} → {rms1:.2f} px).")
            elif self._usable():
                third = (f"Fitting noise with {count} could lower the RMS error by only about "
                         f"{drop:.0%}, yet here it rose ({rms0:.2f} → {rms1:.2f} px).")
            else:
                # ED5-2 / SCI5-3: a held run names each metric and gives no verdict from the mean.
                # One sentence (L3 limit 4): the held-run figures stay as numbers, no verdict.
                # SCI6-4: the bound assumes residuals of one size (Gaussian noise); this run's
                # RMS is set by a few gross residuals, so the bound is conditional here.
                # Round 9 review: say plainly which way the held run's RMS moved.
                change = rms1 / rms0 - 1.0
                # SCI10-3: each measured number carries the tag where it sits, not only at the end
                moved = (f"rose from {rms0:.2f} to {rms1:.2f} px (about {change:.0%}, not verified)"
                         if change >= 0 else
                         f"fell from {rms0:.2f} to {rms1:.2f} px (about {-change:.0%}, not verified)")
                third = (f"If residuals were one-size noise, {count} could lower the RMS by only "
                         f"about {drop:.0%} (not verified); here it {moved}, and the sanity check "
                         "is not judged.")
        else:
            third = ("Extra freedom lowers the objective the solver minimises, so the photos must "
                     "give far more measurements than there are unknowns.")
        shares = v.jacobian_summary().get("weak_shares") if self.level() >= 3 else None
        if shares:
            # SCI3-5: the rank check (E) and the shares come from different matrices; say so.
            fourth = (f"The rank check (E) finds no free direction for the rig; on unit-free scales "
                      f"the weakest directions are mostly photo poses ({shares['photo poses']:.0%}).")
        else:
            fourth = ("The rank check is numerical: a floating-point threshold in mixed units, "
                      "not a measure of precision.")
        del nullity
        return [first, second, third, fourth]

    def _numbers_trust(self) -> None:
        v, n = self.visual, self.narrative
        rows = [(c.title, c.value, f"{c.line}; {c.guard}" if c.guard else c.line)
                for c in v.checks]
        rows += [("Notice", "", note) for note in v.notices]
        rows += [("Blocking", "", reason) for reason in v.blocking]
        nul = v.nullity or {}
        if nul:
            rows += [("Jacobian rank (numerical)", f"{nul.get('rank')} of {nul.get('free_parameters')}",
                      f"shape {nul.get('jacobian_shape')}; float64 rank threshold "
                      f"{fmt(nul.get('tolerance'))} (machine precision, not a statistic; "
                      "singular values are in mixed units)"),
                     ("Null directions", f"{nul.get('global_nullity')} weakly fixed lens or pose, "
                                         f"{nul.get('point_local_nullity')} one-detection corner",
                      "a corner detected by one camera in one photo may slide along its ray "
                      "(two cameras in that photo fix it); other weak directions are not "
                      "symmetries; " + str(nul.get("rank_tolerance_method", "")))]
        unit = self._length_unit()
        k = 1000.0 if unit == "µm" else 1.0
        for label, lo, hi, kind in noise_rows(v, unit):
            text = _length(lo / k, unit) if lo == hi else \
                f"{_length(lo / k, unit)} to {_length(hi / k, unit)}"
            rows.append((f"Noise check: {label}", text,
                         "expected mean 3D distance from noise alone: √3 × the per-coordinate "
                         "formal precision × √(8/3π)" if kind == "noise" else
                         "a real change sits well above the noise line; the inside change and "
                         "the opposite-face widths do not depend on the alignment"))
        tele = bool(v.telecentric) and all(v.telecentric)
        corners, coordinates = held_corners_copy(v)
        if not corners:
            held = "the printed positions of the corners"  # no count: the run recorded none
        elif coordinates:
            held = (f"the printed positions of {corners} corners ({coordinates} numbers: two "
                    "corners in full and one coordinate of the third)")
        else:
            held = f"the printed positions of {corners} corners"
        rows.append(("Moving everything together",
                     "turn: exact; size: exact" + (" with lenses 1.3× weaker" if tele else "")
                     + ("; sideways shift: nearly exact" if tele else "; sideways shift: exact"),
                     "the same photos come out, so the solver fixes one photo's pose (6 numbers) "
                     f"and {held}, and solves for one specific form; "
                     + ("a telecentric lens's image centre absorbs a sideways shift only up to an "
                        "error of about ε times the target's depth (the telecentricity term), so "
                        "that shift is only nearly exact" if tele else
                        "a sideways shift is an exact move for this lens model")))
        drop = v.freedom_drop()
        if drop is not None:
            held_note = ("; if the residuals were noise of one size (not checked on a held run)"
                         if not self._usable() else "")  # SCI6-4
            rows.append(("Error drop from freedom alone", f"≈ {drop:.1%}",
                         "RMS error. 1 − √((m − p)/m): m measured numbers, p effective unknowns "
                         "(rank); what fitting noise with the extra unknowns could explain"
                         + held_note))
        if self.level() >= 3:
            value, how = v.corner_precision()
            rows.append(("Corner precision (formal, per coordinate)",
                         _length(value, unit) if value is not None else "not computed", how))
            shares = v.jacobian_summary().get("weak_shares")
            if shares:
                rows.append(("Weakest directions (unit-free)",
                             ", ".join(f"{name} {share:.0%}" for name, share in shares.items()),
                             "8 weakest non-null directions of the column-normalised JᵀJ; on "
                             "the raw matrix the lens numbers look weakest only by their units"))
        rows.append(("Status", self.run_context.status, n.status_copy()))
        self._numbers(["Check", "Value", "Evidence"], rows)

    # -- stage 5 -------------------------------------------------------------------------

    def _stage_result(self) -> None:
        v, n = self.visual, self.narrative
        level = self.level()
        before, after = n.phase3_end_px(), n.value("d43_final_px")
        classes = getattr(self._derivation, "point_classes", None)
        _fit, fit_tone = n.fit_answer()
        if not self._usable():  # ED4-1: a held-back run shows no verdict, only its numbers
            chips = [Chip(f"{_px(before)} → {_px(after)} px", "held back: not verified", tone=None)]
        else:
            chips = [Chip(f"{_px(before)} → {_px(after)} px", "mean error, Phase 3 → 4",
                          tone=fit_tone)]
        lens = self._lens_pairs()
        if lens is not None:
            # ED5-3: a held run's chips carry the same "not verified" tag, so none reads as a result.
            chips.append(Chip(f"{es.format_sig(max(lens[0].values()))} → "
                              f"{es.format_sig(max(lens[1].values()))} px",
                              "lens distortion, Phase 3 → 4" if self._usable()
                              else "lens distortion · held back: not verified"))
        elif classes is not None:
            several = len(classes.several_directions)
            total = several + len(classes.one_direction) + len(classes.unseen)
            chips.append(Chip(f"{several} of {total}", "corners well seen (back the shape)",
                              tone="good" if total and several >= 0.8 * total else "warn"))
        self._set_chips(chips)
        self.stage_view.set_caption(self._caption(self._result_caption(before, after, lens)))
        self._numbers_result()
        if not v.available or v.final_errors is None:
            reason = v.reason or v.errors_reason or "the saved result could not be checked"
            self._unavailable_figure("Result", f"No result to compare: {reason}.")
            return
        p3 = v.phase3_camera_mean
        p4 = v.phase4_camera_mean or v.final_errors.per_camera()
        after_by = {c: vals for c, (_i, vals) in v.final_errors.by_camera_image().items()}
        trusted = n.trusted_start()
        before_pairs = (v.start_errors.by_camera_image()
                        if v.start_errors is not None and trusted else None)
        before_by = ({c: vals for c, (_i, vals) in before_pairs.items()}
                     if before_pairs is not None else None)
        usable = self._usable()
        held_tag = "" if usable else " (not verified)"
        swap = (dict(v.swap_results()) if trusted and usable and before is not None
                and after is not None else {})
        tele = (bool(v.telecentricity_p4) and bool(v.telecentricity_p3)
                and self.visual.mm_per_unit is not None)

        def per_camera(ax: Any) -> None:
            if p3 is not None:
                es.paired_bars_by_camera(ax, p3, p4, labels=("Phase 3", "Phase 4"),
                                         ylabel="mean error")
                top = max([x for x in list(p3.values()) + list(p4.values()) if np.isfinite(x)]
                          + [1e-12])
                ax.set_ylim(0, top * 1.32)
                legend = ax.get_legend()
                if legend is not None:  # inside the headroom, clear of the title band
                    legend.set_loc("upper right")
                    legend.set_bbox_to_anchor((1.0, 1.0), transform=ax.transAxes)
                es.panel_title(ax, "A", "Per camera: Phase 3 → 4" if usable
                               else "Per camera: Phase 3 and 4 (not verified)",
                               "mean over its corners · hatched: Phase 3 · solid: Phase 4")
            else:
                es.bars_by_camera(ax, p4, ylabel="mean error", unit="px", value_format="{:.2f}")
                es.panel_title(ax, "A", "Per camera after Phase 4" + held_tag,
                               "Phase 3's values are not shown: its start is unconfirmed")

        def lenses(ax: Any, letter: str = "B") -> None:
            # ED6-1: a held run names the panel for what it shows, not for a change it may not have.
            title = ("The lenses changed too" if usable
                     else "Lens distortion: Phase 3 and 4 (not verified)")
            if lens is None:
                es.empty_panel(ax, "The two phases' lenses could not both be read, so they "
                                   "are not compared.")
                es.panel_title(ax, letter, title, None)
                return
            # Round 9 review: one name for the quantity, glossed as a displacement in px.
            draw_log_pairs(ax, lens[0], lens[1], "lens distortion", "px")
            es.panel_title(ax, letter, title,
                           f"{LENS_GLOSS} · hatched: Phase 3 · solid: Phase 4")

        def swapped(ax: Any, letter: str = "C") -> None:
            draw_swap(ax, before, after, swap, self._theme())
            es.panel_title(ax, letter, "Swap one part back",
                           "grey: one part put back · coloured: real")

        def photos(ax: Any, letter: str) -> None:
            if before_by is not None:
                log_strip_pairs(ax, before_by, after_by, self._theme())
                es.panel_title(ax, letter, "Every photo, before → after" if usable
                               else "Every photo (not verified)",
                               "pale: Phase 3 · solid: Phase 4 · bar: median")
            else:
                es.strip_plot_by_camera(ax, after_by, unit="px", log=True,
                                        ylabel=es.error_label("mean", per="photo", unit=False,
                                                              short=True))
                es.panel_title(ax, letter, "Every photo after Phase 4" + held_tag,
                               "one dot per photo and camera · log scale")

        def depth_term(ax: Any, letter: str) -> None:
            draw_signed_pairs(ax, v.telecentricity_p3, v.telecentricity_p4, self._theme())
            # 1/m and % per cm are the same number: eps [1/m] x 1 cm x 100 %.
            ax.set_ylabel("ε, size change (% per cm)")
            es.panel_title(ax, letter, "Size change with depth ε" + held_tag,
                           "hatched: Phase 3 · solid: Phase 4")

        def scatter(ax: Any, letter: str) -> None:
            draw_before_after_scatter(ax, before_pairs, v.final_errors.by_camera_image(),
                                      self._theme(), letter, verified=usable)

        def draw(fig: Any, _theme: str) -> None:
            # Panels in teaching order; only those this run can back (no empty panels).
            panels: list[Callable[[Any, str], None]] = [lambda ax, _l: per_camera(ax)]
            if lens is not None:
                panels.append(lenses)
            if swap:
                panels.append(swapped)
            panels.append(photos)
            if tele and level >= 2:
                panels.append(depth_term)
            if level >= 3 and before_pairs is not None:
                panels.append(scatter)
            if level == 1:
                panels = panels[:3]
            top, rest = panels[:3], panels[3:]
            rows = 2 if rest else 1
            if usable:
                gs = fig.add_gridspec(rows, 3, width_ratios=[1.05, 1.0, 1.0])
            else:
                # ED6-1: one visible tag on the figure itself, in its own row above the panels
                # (so no panel can read as a result on its own).
                outer = fig.add_gridspec(2, 1, height_ratios=[0.11, 1.0])
                draw_held_band(fig.add_subplot(outer[0, 0]), _theme,
                               "held back: not verified · these values are for inspection only")
                gs = outer[1, 0].subgridspec(rows, 3, width_ratios=[1.05, 1.0, 1.0])
            slots = {1: [gs[0, :]], 2: [gs[0, 0], gs[0, 1:]],
                     3: [gs[0, 0], gs[0, 1], gs[0, 2]]}[len(top)]
            if rest:
                slots += {1: [gs[1, :]], 2: [gs[1, :2], gs[1, 2]],
                          3: [gs[1, 0], gs[1, 1], gs[1, 2]]}[min(len(rest), 3)]
            for k, (fn, slot) in enumerate(zip(panels, slots)):
                fn(fig.add_subplot(slot), "ABCDEF"[k])
            _lighten_hatches(fig, _theme)
        self.stage_view.set_controls(None)
        self.stage_view.draw_figure(draw)

    def _lens_pairs(self) -> tuple[dict[int, float], dict[int, float]] | None:
        """Each camera's largest lens-distortion bend (px) in Phase 3 and Phase 4, when both
        were read and the start is Phase 3's (ED2-4 / SCI2-1); else None."""
        v = self.visual
        if not (v.distortion_p3 and v.distortion_p4 and self.narrative.trusted_start()):
            return None
        cams = sorted(set(v.distortion_p3) & set(v.distortion_p4))
        if not cams or max(v.distortion_p3[c] for c in cams) <= 0:
            return None
        return ({c: v.distortion_p3[c] for c in cams}, {c: v.distortion_p4[c] for c in cams})

    def _result_caption(self, before: Any, after: Any,
                        lens: tuple[dict[int, float], dict[int, float]] | None) -> list[str]:
        """ED2-1: say what was freed (the corners) and what kept being refined (lenses, camera
        turns); credit the drop to both, with the lens-vs-shape trade-off (ED2-4 / SCI2-1)."""
        v, n = self.visual, self.narrative
        direction = n.fit_direction()
        usable = self._usable()
        lead = {"lower": f"The mean error fell from {_px(before)} to {_px(after)} px with corners, "
                         "lenses and cameras refined together.",
                "higher": "The mean error went up instead of down, so do not rely on this run.",
                "same": "The mean error did not go down, so do not rely on this run."}.get(
            direction, "The fit comparison cannot be checked for this run.")
        if not usable:  # ED4-1 / SCI4-2 / SCI5-3: a held run's numbers are not a verified improvement
            lead = (f"Held back, not verified: the mean error reads {_px(before)} → {_px(after)} px "
                    "for inspection only, and these numbers do not show that the calibration improved.")
        swap = v.swap_results() if n.trusted_start() and usable else {}
        if direction == "lower" and lens is not None:
            if usable:  # the lens-for-shape reading is checked only on a run that passed
                # ED8-3: "lens distortion" is glossed here, once, and used for the same thing after.
                # ED9-3: says what was measured (the fits bent the photos while the cube shape was
                # drawn), not why; the cause is not tested here.
                second = (f"Phase 3's lens distortion ({LENS_GLOSS}) bent photos up to "
                          f"{es.format_sig(max(lens[0].values()))} px while the cube shape was drawn; "
                          f"re-fitted, they bent {es.format_sig(max(lens[1].values()))} px (B).")
            else:
                second = (f"Phase 3's lens distortion ({LENS_GLOSS}) reached "
                          f"{es.format_sig(max(lens[0].values()))} px, "
                          f"{es.format_sig(max(lens[1].values()))} px with the shape free (B).")
        else:
            second = n.fit_credit()
        if direction == "lower" and {"lens_swap", "shape_swap"} <= set(swap):
            # SCI7-1: each swap puts one part back to Phase 3 and keeps the other at Phase 4.
            # Neither alone reproduces Phase 4, and the shape-back value is above Phase 3's own.
            # Round 9 review: the photo poses are re-fitted but the solver is not re-run, so each value
            # is an upper bound on the best that part could do; it is not a measured minimum.
            third = (f"One part back to Phase 3, photo poses re-fitted (C): lenses at most "
                     f"{es.format_number(swap['lens_swap'])} px, shape at most "
                     f"{es.format_number(swap['shape_swap'])} px, against Phase 4's "
                     f"{es.format_number(after)} px.")
        else:
            # SCI5-4: extra freedom lowers the objective the solver minimises, not every error.
            third = ("Extra freedom lowers the solver's objective, so stage 4 checks the shape "
                     "change against its noise.")
        p3e, p4e = v.telecentricity_p3, v.telecentricity_p4
        if p3e and p4e and v.mm_per_unit is not None:
            lo3, hi3 = min(p3e.values()), max(p3e.values())
            lo4, hi4 = min(p4e.values()), max(p4e.values())
            flipped = (hi3 < 0 < lo4) or (lo3 > 0 > hi4)
            letter = "ABCDEF"[2 + (lens is not None) + bool(swap)]  # the ε panel
            if usable:
                fourth = (f"ε {'flipped sign' if flipped else 'changed'} too ({letter}): a bigger cube "
                          "looks like a lens that magnifies more with depth.")
            else:
                fourth = f"On this held run, ε ({letter}) is shown for inspection only."
        else:
            fourth = ("Hatched bars are Phase 3, solid bars Phase 4; one dot per photo, so a "
                      "single bad photo stands out.")
        if self.level() == 1 and direction == "lower" and {"lens_swap", "shape_swap"} <= set(swap):
            # ED10-1: the L1 caption holds two sentences, so the swap test (C), which is the
            # stage's central check, takes the second place; panel B still shows the lens change.
            # Upper bounds, as on L2 and L3 (photo poses re-fitted, solver not re-run).
            swap_line = (f"Put one part back to Phase 3 and the error is at most "
                         f"{es.format_number(swap['lens_swap'])} px (lenses) or "
                         f"{es.format_number(swap['shape_swap'])} px (shape), against "
                         f"{es.format_number(after)} px (C).")
            return [lead, swap_line]
        return [lead, second, third, fourth]

    def _numbers_result(self) -> None:
        n, v = self.narrative, self.visual
        if self.run_context.status == "failed":
            self._numbers(["Quantity", "Value", "Meaning"],
                          [("Phase 4 result", "failed", n.status_copy())])
            return
        before = n.phase3_end_px()
        start = (self.metadata.get("diagnostics") or {}).get("D4.3_initial_euclid_px")
        after = n.value("d43_final_px")
        if self.level() >= 2:
            rows = [("Before", fmt(before), "Phase 3 D3.6 from its verified state source"),
                    ("Phase 4 start", fmt(start), "D4.3, px"),
                    ("Phase 4 result", fmt(after), "D4.3, px; recalculated from saved results")]
        else:
            rows = [("Before", fmt(before), "Phase 3's final average error, px"),
                    ("Phase 4 start", fmt(start), "where Phase 4 began, average error, px"),
                    ("Phase 4 result", fmt(after),
                     "Phase 4's final average error, px; recalculated from saved results")]
        p3, p4 = v.phase3_camera_mean or {}, v.phase4_camera_mean or {}
        for c, name in enumerate(v.camera_names):
            rows.append((f"Camera {c + 1} ({name})", f"{fmt(p3.get(c))} → {fmt(p4.get(c))} px",
                         "D3.12 → D4.12 mean error" if self.level() >= 2
                         else "mean error, Phase 3 → Phase 4"))
        if self.level() >= 2 and v.telecentricity_p4:
            for c, value in sorted(v.telecentricity_p4.items()):
                rows.append((f"Camera {c + 1} ε", f"{fmt(v.telecentricity_p3.get(c))} → "
                             f"{fmt(value)} % per cm",
                             "size change with depth, signed (Phase 3 → 4; % per cm is the same "
                             "number as 1/m): it changed together with the shape, so the two "
                             "are linked"))
        lens = self._lens_pairs()
        if lens is not None:
            for c in sorted(lens[0]):
                rows.append((f"Camera {c + 1} lens distortion", f"{fmt(lens[0][c])} → "
                             f"{fmt(lens[1][c])} px",
                             f"{LENS_GLOSS}, largest in the photo (Phase 3 → 4), measured as "
                             "Phase 3's Lens tuning does, about the point the lens model distorts "
                             "around"))
        if not self._usable():
            rows.extend(self._held_spread_rows())  # SCI5-3: what the mean alone hides
        if n.trusted_start() and self._usable():
            # SCI7-1: one frame (the solver's own), each swap re-fitted from two pose starts.
            names = {"lens_swap": "Lenses back to Phase 3, shape as Phase 4",
                     "shape_swap": "Shape back to Phase 3, lenses as Phase 4"}
            swap = v.swap_results()
            for key in ("lens_swap", "shape_swap"):
                if key not in swap:
                    continue
                starts = [f"{fmt(swap[k])} from {w}" for k, w in
                          ((f"{key[:-5]}_p4poses", "Phase 4's poses"),
                           (f"{key[:-5]}_p3poses", "Phase 3's poses")) if k in swap]
                rows.append((names[key], f"{fmt(swap[key])} px",
                             "upper bound: the best the photo poses alone reach with that part put "
                             "back, from both starts (" + "; ".join(starts) + "), the lower value "
                             "shown; the solver is not re-run"))
        rows.append(("Attribution", "", n.fit_credit()))
        rows.append(("Decision", "", n.decision_copy()))
        if not self._usable():  # ED4-1: the Numbers drawer must not read as a verdict either
            rows.insert(0, ("Verified", "no", "held back: these numbers are not a calibration"))
        if self.level() >= 2:
            rows.append(("Loss", "", n.loss_detail()))
        self._numbers(["Quantity", "Value", "Meaning"], rows)


    # -- stage 6 -------------------------------------------------------------------------

    def _held_spread_rows(self) -> list[tuple[str, str, str]]:
        """Neutral spread facts for a held run (SCI5-3): the median and how many corner
        observations are 100 px or more, with their share of the sum of squared errors."""
        rows: list[tuple[str, str, str]] = []
        for label, table in (("Median error, Phase 4 start", self.visual.start_errors),
                             ("Median error, Phase 4 result", self.visual.final_errors)):
            norms = np.asarray(getattr(table, "norms", ()), dtype=float)
            if norms.size:
                rows.append((label, f"{float(np.median(norms)):.2f} px",
                             "per corner observation; the mean is pulled up by the largest errors"))
        final = np.asarray(getattr(self.visual.final_errors, "norms", ()), dtype=float)
        if final.size and float(np.sum(final ** 2)) > 0.0:
            big = final >= 100.0
            share = float(np.sum(final[big] ** 2) / np.sum(final ** 2))
            rows.append(("Observations at 100 px or more", f"{int(big.sum()):,} of {final.size:,}",
                         f"they carry {share:.1%} of the sum of squared errors, which the RMS follows"))
        return rows

    def _usable(self) -> bool:
        """The run passed its gate: its numbers may be quoted as the calibration's."""
        return self.run_context.status == "complete" and not self.visual.blocking

    def _held(self) -> bool:
        """A held-back run (gate not passed, not failed): its numbers carry the held tag."""
        return not self._usable() and self.run_context.status != "failed"

    def _closing_um(self) -> tuple[float | None, float | None]:
        """(median µm per pixel over the cameras, mean error in µm); None without units."""
        summary = self.visual.closing_results()
        after = self.narrative.value("d43_final_px")
        if summary is None or not summary.um_per_px:
            return None, None
        per_px = [x for x in summary.um_per_px.values() if np.isfinite(x)]
        if not per_px:
            return None, None
        um = float(np.median(per_px))
        return um, (float(after) * um if after is not None else None)

    def _stage_closing(self) -> None:
        """What the calibration is for (ED3-2 / ED-13): a pixel is a ray, two cameras' rays
        meet at a corner, and the error is a length on the target (µm)."""
        v, n = self.visual, self.narrative
        level = self.level()
        usable = self._usable()
        um, error_um = self._closing_um()
        summary = v.closing_results()
        after = n.value("d43_final_px")
        if usable and um is not None and after is not None:
            self._set_chips([Chip(f"{_px(after)} px ≈ {error_um:.0f} µm", "mean error on the target",
                                  tone="good"),
                             Chip(f"{um:.0f} µm/px", "x scale, median of the cameras")])
        else:
            self._set_chips([Chip("held back" if not usable else "no µm",
                                  "the precision is not quoted", tone="bad" if not usable else None)])
        sentences = self._closing_caption(usable, um, error_um, summary)
        self.stage_view.set_caption(self._caption(sentences))
        self._numbers_closing(usable, um, error_um, summary)

        def draw(fig: Any, theme: str) -> None:
            if not usable:
                # ED7-5: the held run keeps the ray picture (tagged "not quoted") beside the
                # reasons card, side by side as on a passed run, so the stage teaches the idea
                # and does not leave the lower right of the stage empty.
                gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
                draw_corner_rays(fig.add_subplot(gs[0, 0]), summary, theme, "A", held=True)
                draw_closing_held(fig.add_subplot(gs[0, 1]), theme, v.blocking, len(v.excluded),
                                  letter="B", wrap=62)
                return
            if level == 1:
                gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
                top = [gs[0, 0], gs[0, 1]]
                rest = []
            else:
                gs = fig.add_gridspec(2, 2, height_ratios=[1.75, 0.6], width_ratios=[1.25, 1.0])
                top = [gs[0, 0], gs[0, 1]]
                rest = [gs[1, :]]
            draw_corner_rays(fig.add_subplot(top[0]), summary, theme, "A")
            draw_precision_bars(fig.add_subplot(top[1]), n, v, um, theme, "B")
            if rest:
                draw_redo_card(fig.add_subplot(rest[0]), v, theme, "C")
        self.stage_view.set_controls(None)
        self.stage_view.draw_figure(draw)

    def _closing_caption(self, usable: bool, um: float | None, error_um: float | None,
                         summary: Any) -> list[str]:
        if not usable:  # the run's own warning comes first (see _caption)
            return ["A pixel still fixes a ray through the lens model, but a corner's place needs a "
                    "run that passed the gate, so no precision is quoted."]
        ray = ("Each camera's pixel, once distortion is removed, fixes a ray: the corner must "
               "lie somewhere on that line.")
        # ED4-3 / SCI4-4: one number here, and what it is. The 0.30 px is a mean reprojection
        # residual, so it is a typical image-plane error expressed at the target, not the
        # precision of a measured position (the formal precision is in stage 4).
        if um is not None and error_um is not None:
            # Round 9 review (ED9-2 / SCI9-3): one mean, one scale, the conversion stated once.
            precision = (f"The mean error of {_px(self.narrative.value('d43_final_px'))} px × "
                         f"{um:.0f} µm per pixel (median x scale) is about {error_um:.0f} µm on "
                         "the target: an image-plane error, not a precision.")
        else:
            precision = ("This run has no recorded units, so its error is given in pixels only.")
        meet = ""
        if summary is not None and summary.gap_um_median is not None and summary.n_pairs:
            meet = (f"Rays of two cameras miss each other by {summary.gap_um_median:.1f} µm "
                    f"(median over {summary.n_pairs:,} camera pairs): a check on the fitted model.")
        file = ("The calibration file is saved with this run; Numbers gives its path."
                if summary is not None and summary.file else "")
        return [ray, precision, meet, file]

    def _numbers_closing(self, usable: bool, um: float | None, error_um: float | None,
                         summary: Any) -> None:
        v, n = self.visual, self.narrative
        rows: list[tuple[str, str, str]] = []
        if summary is not None and summary.file:
            rows.append(("Calibration file", summary.file,
                         "saved with this run (the Phase 4 result); it is not changed here"))
        if not usable:
            rows.append(("Status", self.run_context.status, n.status_copy()))
            self._numbers(["Quantity", "Value", "Meaning"], rows)
            return
        if summary is not None and summary.um_per_px:
            per = ", ".join(f"{self._cam_name(c)} {x:.1f}" for c, x in
                            sorted(summary.um_per_px.items()) if np.isfinite(x))
            rows.append(("x scale, µm per pixel (each camera)", per,
                         "the saved cameras' own x scales, not a pitch-based estimate; the closing "
                         "shows their median; telecentric: 10⁶ / focal length in pixels per metre; "
                         "the same at any distance"))
        # ED7-4: stated for any run; the figures come from this run's saved cameras and detections.
        if summary is not None:
            rows.append(("Source of these figures", "this run's saved cameras and detections",
                         "these figures are computed from them; nothing is copied from another "
                         "run"))
        after = n.value("d43_final_px")
        if after is not None and error_um is not None:
            rows.append(("Mean error on the target", f"{_px(after)} px ≈ {error_um:.1f} µm",
                         "mean reprojection error × µm per pixel (median camera); a typical "
                         "image-plane error at the target, not a measurement precision"))
        sd3, _mean3 = v.noise_3d()
        if sd3 is not None:
            rows.append(("3D precision of a corner (formal, 1σ)", f"≈ {sd3 * 1000:.0f} µm",
                         "from the corners' formal precision, the same figure the noise check uses"))
        if summary is not None and summary.gap_um_median is not None:
            rows.append(("Rays of two cameras, one corner",
                         f"median miss {summary.gap_um_median:.1f} µm",
                         f"over {summary.n_pairs:,} camera pairs (two cameras seeing one corner "
                         "in one photo); "
                         f"median angle between their rays {summary.angle_deg_median:.0f}°"))
        if summary is not None and summary.example is not None:
            e = summary.example
            rows.append(("Example corner", f"photo {e.photo + 1}, {self._cam_name(e.cameras[0])} "
                         f"and {self._cam_name(e.cameras[1])}",
                         f"rays {e.angle_deg:.0f}° apart, {e.gap * summary.um_per_unit:.1f} µm "
                         "apart at closest; pixels (" + ", ".join(
                             f"{x:.1f}, {y:.1f}" for x, y in e.pixels) + ")"))
        rows.append(("Status", self.run_context.status, n.status_copy()))
        self._numbers(["Quantity", "Value", "Meaning"], rows)

    def _cam_name(self, index: int) -> str:
        names = self.visual.camera_names
        return es.camera_title(index, names[index] if 0 <= index < len(names) else None,
                               max_name=14)

def create_education_phase_view(context: Any, parent: QWidget | None = None) -> Phase4View:
    """Entry point discovered lazily by the shared education dialog."""
    return Phase4View(context, parent)


__all__ = ["Phase4View", "SplitVisual", "MorphBar", "create_education_phase_view",
           "display_rotation", "draw_gate_cards", "draw_coverage", "draw_excluded_cards",
           "draw_unknowns", "draw_gauge_strip", "draw_rank_check", "draw_face_slice",
           "draw_face_bars", "log_strip_pairs", "MORPH_POSITIONS"]
