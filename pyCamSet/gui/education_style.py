"""One visual language for every education-mode figure (palette, figure style, helpers).

Colours come from the GUI's own themes (``gui/theme.py`` THEME_TOKENS: Light, Dark, Sepia):
figure/axes background = theme ``surface``, text = ``text`` /
``text_muted``, detections and highlights = ``accent``, good/warn/bad = success/warning/danger,
rejected/unused = muted at :data:`REJECTED_ALPHA`. See :func:`stage_tokens`.
The figures share one structure:

* every camera has one fixed colour (:func:`camera_colour`) and a numbered badge
  (:func:`camera_badge_label`, 1-based) in *every* figure, badge and legend;
* continuous values use viridis with a labelled colour bar in px (:func:`colourbar_px`);
* one bold 13 pt title + one 10 pt muted subtitle per panel (:func:`panel_title`);
* per-image strip plots per camera (:func:`strip_plot_by_camera`).

Typical use inside a phase (``EducationStageView.draw_figure`` does the first two lines)::

    with matplotlib.rc_context(figure_style(theme)):
        ax = figure.add_subplot()
        image_with_points(ax, photo, corners, camera=3)
        panel_title(ax, "A", "Camera image", "camera 4, 63 corners")

Camera *indices* are 0-based everywhere in this API (as stored in the run); only the
badge text is 1-based. The helpers read the theme back from the axes background, so
they need no theme argument once the figure was styled.

Shared rules added after review round 1 (2026-10-08), one place for all four phases:

* **Figure colours vs camera colours** (UX-8): ``stage_tokens`` keeps the theme's accent,
  link, warn, bad and good unless one is closer than :data:`MIN_CAMERA_DISTANCE` (CIE76
  Delta E) to any camera colour or has < :data:`MIN_STAGE_CONTRAST` against the surface; then
  a fixed fall-back is used (Sepia accent -> teal, Light/Sepia warn -> ochre, Dark warn ->
  yellow). Never draw meaning in a camera hue.
* **Camera outline rule** (UX-7): a camera mark keeps its fill; below
  :data:`CAMERA_OUTLINE_CONTRAST` it gets an edge in ``stage_tokens()["outline"]``
  (:func:`camera_mark_kwargs`, :func:`camera_line_effects`).
* **Continuous colour** (UX-6): :func:`error_cmap` (viridis, low end clipped on Dark) and
  :func:`value_dot_kwargs` (thin edge on value-coloured dots).
* **Reprojection error vocabulary** (ED-19): :func:`error_label` and one log colour scale
  :func:`error_scale` (0.1-100 px, :data:`ERROR_SCALE_PX`) for every phase.
* **Wording**: photos numbered from 1 (:func:`photo_label`, SCI-8), :func:`camera_title`,
  :func:`format_number` and friends, a :data:`GLOSSARY`, caption limits
  (:func:`caption_problems`).
* **Layout** (UX-1, UX-13): :class:`EduLayoutEngine` / :func:`align_panel_titles` (one title
  baseline per row, elided titles, value labels and badges that never collide) and
  :func:`text_collisions` (the checker tests and review screenshots use).

Round-3 rules (2026-10-09):

* **Colour maps** (UX3-3): viridis (:func:`error_cmap`) is reserved for reprojection error and
  displacements in px / µm / mm. Counts, density, coupling and shares use the one-hue
  :func:`amount_cmap` (:func:`amount_scale`); signed quantities the diverging
  :func:`signed_cmap` (:func:`signed_scale`). :func:`colour_map_misuse` checks it.
* **Limits** (UX3-11): :func:`limit_line` / :func:`limit_style` draw a limit in neutral
  dashes and switch to red only when a value crosses it; playback cursors use
  :func:`cursor_style` (accent).
* **Cards** (UX3-1): write findings on a card with :func:`card_text` (wrapped/shrunk to fit at
  every layout); :func:`card_overflows` reports text running out of its card.
* **Captions** (UX3-2): line limits hold at every :data:`SUPPORTED_DIALOG_WIDTHS` width and a
  caption never ends in a one-word line (:func:`caption_problems`).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------
# Palette

REJECTED_ALPHA = 0.45     #: alpha for rejected / unused items (drawn in the theme's muted)
WHITE = "#FFFFFF"
CONTINUOUS_CMAP = "viridis"

#: Fixed per-camera colours (poster order); cameras beyond 8 continue with tab10.
CAMERA_PALETTE: tuple[str, ...] = (
    "#D4B483", "#8B5A3C", "#D98BA6", "#F28C28", "#A9B8C9", "#3B4252", "#A48BBF", "#9C9283",
)
_TAB10: tuple[str, ...] = (
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
)


#: Preferred figure fonts (first installed one wins; DejaVu Sans ships with matplotlib).
FONT_FAMILIES = ("Segoe UI", "Arial", "Liberation Sans", "DejaVu Sans")
#: The font caption line budgets (:data:`CAPTION_MAX_LINES`) were set in; captions are
#: only counted in lines where they render in it.
CAPTION_BUDGET_FONT = FONT_FAMILIES[0]
TITLE_SIZE = 13
SUBTITLE_SIZE = 10
TICK_SIZE = 9
LABEL_SIZE = 10


#: Per-theme substitute fills (UX2-1, round-2 review): a camera whose poster colour has
#: < :data:`CAMERA_OUTLINE_CONTRAST` against a theme's figure background is drawn in ONE
#: substitute fill on that theme, on every kind of mark (badge, bar, dot, raster cell, line,
#: frame, 3D glyph), never hollow, never white. Each substitute keeps its camera's hue family,
#: has >= 2.3:1 contrast on the stage and stays >= 25 CIE76 Delta E from every other camera
#: colour (tests check all three). Only badges also get the thin outline ring.
CAMERA_THEME_FILLS: dict[str, dict[int, str]] = {
    "Light": {0: "#BE9F67", 4: "#8BA7BD"},   # camera 1 (sand), camera 5 (blue-grey)
    "Sepia": {0: "#BE9F67", 4: "#8BA7BD"},
    "Dark": {5: "#5D7A97"},                   # camera 6 (slate; UX4-5: lifted to 3:1 on Dark)
}


def camera_colour(index: int, theme: Any = None) -> str:
    """Hex colour of camera ``index`` (0-based): poster palette, then tab10, cycling.

    ``theme`` (a theme name, an axes or a figure): the colour to *draw* with on that theme,
    i.e. :func:`camera_fill` (the substitute fill for low-contrast cameras, UX2-1). Without
    ``theme`` the poster identity colour is returned (legends of the palette itself, tests).
    Always pass the axes when drawing: ``camera_colour(cam, ax)``.
    """
    index = int(index)
    if index < 0:
        raise ValueError("camera index must be >= 0")
    if theme is not None:
        return camera_fill(index, theme)
    if index < len(CAMERA_PALETTE):
        return CAMERA_PALETTE[index]
    return _TAB10[(index - len(CAMERA_PALETTE)) % len(_TAB10)]


def camera_fill(index: int, theme: Any = None) -> str:
    """The one colour every mark of camera ``index`` is drawn with on ``theme`` (a theme
    name, an axes, a figure or None = the active theme): the poster colour, or its theme
    substitute (:data:`CAMERA_THEME_FILLS`) when the poster colour would melt into the
    background. Same rule for badges, bars, dots, raster cells, lines, frames and 3D glyphs."""
    index = int(index)
    if index < 0:
        raise ValueError("camera index must be >= 0")
    fills = CAMERA_THEME_FILLS.get(_theme_of(theme), {})
    return fills.get(index) or camera_colour(index)


def camera_badge_label(index: int) -> str:
    """Badge text for camera ``index`` (0-based): the 1-based number, e.g. ``0 -> "1"``."""
    return str(int(index) + 1)


def _rgb(colour: str) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb
    return to_rgb(colour)


def relative_luminance(colour: str) -> float:
    """WCAG relative luminance of a colour (0 black .. 1 white)."""
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in _rgb(colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def badge_text_colour(index: int, theme: Any = None) -> str:
    """Black or white number on camera ``index``'s badge, whichever reads better (poster
    rule); ``theme`` uses that theme's fill (:func:`camera_fill`)."""
    base = "#000000" if relative_luminance(camera_colour(index)) > 0.40 else WHITE
    if theme is None:
        return base
    fill = camera_fill(index, theme)  # a substitute keeps the poster's number colour if legible
    if contrast(base, fill) >= 3.0:
        return base
    return WHITE if base == "#000000" else "#000000"


def contrast(foreground: str, background: str) -> float:
    """WCAG contrast ratio between two colours (1 .. 21)."""
    a, b = sorted((relative_luminance(foreground), relative_luminance(background)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def theme_names() -> tuple[str, ...]:
    """The GUI's themes (``THEME_TOKENS`` keys: Light, Dark, Sepia)."""
    from pyCamSet.gui.theme import THEME_TOKENS
    return tuple(THEME_TOKENS)


def resolve_theme(theme_name: str | None) -> str:
    """A valid ``THEME_TOKENS`` name (case-insensitive; ``None`` -> active app theme; unknown -> Light)."""
    from pyCamSet.gui.theme import THEME_TOKENS
    if theme_name is None:
        return active_theme_name()
    for name in THEME_TOKENS:
        if name.casefold() == str(theme_name or "").casefold():
            return name
    return "Light"


def active_theme_name() -> str:
    """The running application's theme (Qt property ``pycamsetTheme``), else ``Light``.

    Safe without Qt or without a QApplication.
    """
    import sys
    widgets = sys.modules.get("PySide6.QtWidgets")
    app = widgets.QApplication.instance() if widgets is not None else None
    name = app.property("pycamsetTheme") if app is not None else None
    return resolve_theme(name) if name else "Light"


def colour_distance(a: str, b: str) -> float:
    """CIE76 colour difference (Delta E in CIELAB, D65) between two colours.

    About 2.3 is a just-noticeable difference; :data:`MIN_CAMERA_DISTANCE` (25) is "clearly
    a different hue family" at a glance.
    """
    return float(np.linalg.norm(_lab(a) - _lab(b)))


def _lab(colour: str) -> np.ndarray:
    rgb = np.asarray(_rgb(colour), dtype=float)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = m @ lin / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


#: Figure-colour rule (UX-8, round-1 review): every non-camera colour a figure uses for
#: meaning (accent = detections/highlights, link = matches/rays, warn, bad, good) must be
#: at least this CIE76 Delta E from *every* fixed camera colour, so that a learner never reads
#: a warning, a detection or a ray as "camera 2" or "camera 4" ...
MIN_CAMERA_DISTANCE = 25.0
#: ... and at least this WCAG contrast against the figure background (theme surface).
MIN_STAGE_CONTRAST = 3.0
#: Per role: the theme token tried first, then fixed fall-backs in order. The first colour
#: that passes both rules wins. Light/Dark keep their own accent; Sepia's brown accent
#: (= camera 2) becomes teal; the orange warnings (= camera 4) become ochre (light themes)
#: or yellow (Dark).
_FIGURE_ROLE_CANDIDATES: dict[str, tuple[str, tuple[str, ...]]] = {
    "accent": ("accent", ("#0F766E", "#1D4ED8", "#2563EB", "#22D3EE")),
    "link": ("accent_hover", ("#1D4ED8", "#0E7490", "#7DD3FC", "#2563EB")),
    "warn": ("warning", ("#A16207", "#FACC15", "#CA8A04")),
    "bad": ("danger", ("#B91C1C", "#F87171", "#DC2626")),
    "good": ("success", ("#15803D", "#34D399", "#4E6B31")),
}


def _figure_role_colour(tok: Mapping[str, str], role: str) -> str:
    token, fallbacks = _FIGURE_ROLE_CANDIDATES[role]
    candidates = (tok[token],) + fallbacks
    best, best_score = candidates[0], -1.0
    for colour in candidates:
        distance = min(colour_distance(colour, cam) for cam in CAMERA_PALETTE)
        ok = distance >= MIN_CAMERA_DISTANCE and contrast(colour, tok["surface"]) >= \
            MIN_STAGE_CONTRAST
        if ok:
            return colour
        score = min(distance / MIN_CAMERA_DISTANCE,
                    contrast(colour, tok["surface"]) / MIN_STAGE_CONTRAST)
        if score > best_score:
            best, best_score = colour, score
    return best


def stage_tokens(theme_name: str | None) -> dict[str, str]:
    """Figure colours for a GUI theme, all derived from ``gui/theme.py`` THEME_TOKENS.

    Keys: ``stage`` (figure/axes background = theme surface), ``text``, ``muted``,
    ``title`` (= text), ``accent`` (detections, highlights), ``link`` (matches, rays),
    ``good``/``warn``/``bad``, ``rejected`` (muted; draw at :data:`REJECTED_ALPHA`),
    ``grid`` (border), ``spine`` (border_strong), ``badge_edge`` (ring for highlighted
    badges), ``outline`` (edge for camera marks that would melt into the background, see
    :func:`camera_edge`), ``point_edge`` (thin edge on value-coloured dots), ``theme``.

    ``accent``, ``link``, ``warn``, ``bad`` and ``good`` are the theme's own tokens unless
    one collides with a camera colour or lacks contrast; then a fixed fall-back is used
    (rule: :data:`MIN_CAMERA_DISTANCE`, :data:`MIN_STAGE_CONTRAST`). The raw theme tokens
    stay available as ``ui_accent`` and ``ui_warn`` (Qt chrome only, never data).
    """
    name = resolve_theme(theme_name)
    cached = _TOKEN_CACHE.get(name)
    if cached is not None:
        return dict(cached)
    from pyCamSet.gui.theme import THEME_TOKENS
    tok = THEME_TOKENS[name]
    dark = relative_luminance(tok["surface"]) < 0.2
    tokens = {
        "theme": name, "stage": tok["surface"], "text": tok["text"],
        "muted": tok["text_muted"], "title": tok["text"],
        "accent": _figure_role_colour(tok, "accent"), "link": _figure_role_colour(tok, "link"),
        "good": _figure_role_colour(tok, "good"), "warn": _figure_role_colour(tok, "warn"),
        "bad": _figure_role_colour(tok, "bad"), "rejected": tok["text_muted"],
        "grid": tok["border"], "spine": tok["border_strong"], "badge_edge": tok["text"],
        "outline": tok["text_muted"], "point_edge": "#FFFFFF" if dark else "#1E2432",
        "ui_accent": tok["accent"], "ui_warn": tok["warning"],
    }
    _TOKEN_CACHE[name] = tokens
    return dict(tokens)


_TOKEN_CACHE: dict[str, dict[str, str]] = {}


def theme_colour(theme_name: str | None, role: str) -> str:
    """One colour role from :func:`stage_tokens`, e.g. ``theme_colour("Dark", "accent")``."""
    return stage_tokens(theme_name)[role]


_FONT_CACHE: list[str] = []


def _installed_fonts() -> list[str]:
    if not _FONT_CACHE:
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        _FONT_CACHE.extend([f for f in FONT_FAMILIES if f in names] or ["DejaVu Sans"])
    return list(_FONT_CACHE)


def figure_style(theme_name: str | None = None) -> dict[str, Any]:
    """matplotlib rc dict for the poster look; use as ``with rc_context(figure_style(theme)):``.

    ``theme_name`` is a THEME_TOKENS name (Light, Dark, Sepia); ``None`` = the running app's
    theme (:func:`active_theme_name`). Create the axes *inside* the context (rc values are
    read when artists are made).
    Constrained layout is on, top/right spines are off, grids are muted and off by default.
    """
    t = stage_tokens(theme_name)
    return {
        "figure.facecolor": t["stage"], "axes.facecolor": t["stage"],
        "savefig.facecolor": t["stage"], "figure.edgecolor": t["stage"],
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.06, "figure.constrained_layout.w_pad": 0.08,
        "figure.constrained_layout.hspace": 0.06, "figure.constrained_layout.wspace": 0.06,
        # Concrete families, not "sans-serif": a generic family is resolved at *draw* time,
        # outside this rc context, and would fall back to DejaVu Sans.
        "font.family": _installed_fonts(),
        "font.sans-serif": _installed_fonts(),
        "font.size": LABEL_SIZE,
        "text.color": t["text"], "axes.labelcolor": t["text"],
        "axes.titlecolor": t["title"], "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.labelsize": LABEL_SIZE, "axes.edgecolor": t["spine"], "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": t["grid"], "grid.linewidth": 0.8, "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "axes.prop_cycle": __import__("cycler").cycler(color=list(CAMERA_PALETTE)),
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["text"], "ytick.labelcolor": t["text"],
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "legend.fontsize": TICK_SIZE, "legend.frameon": False, "legend.labelcolor": t["text"],
        "image.cmap": error_cmap(t["theme"]), "lines.linewidth": 1.6, "lines.markersize": 5,
        "scatter.edgecolors": "none",
    }


def axes_theme(ax: Any) -> str:
    """The GUI theme an axes was styled for, read back from its background colour.

    Helpers use it so they need no theme argument. A figure may also carry an explicit
    ``figure._edu_theme`` (set by :func:`apply_figure_style` and the stage view).
    """
    explicit = getattr(ax.figure, "_edu_theme", None)
    if explicit:
        return resolve_theme(explicit)
    from matplotlib.colors import to_hex
    from pyCamSet.gui.theme import THEME_TOKENS
    face = ax.get_facecolor() if ax.get_facecolor()[3] > 0 else ax.figure.get_facecolor()
    hex_face = to_hex(face[:3]).casefold()
    for name, tok in THEME_TOKENS.items():
        if tok["surface"].casefold() == hex_face:
            return name
    return "Dark" if relative_luminance(hex_face) < 0.2 else "Light"


def is_dark_axes(ax: Any) -> bool:
    """True when ``ax`` sits on a dark background."""
    return relative_luminance(_tokens_for(ax)["stage"]) < 0.2


def _tokens_for(ax: Any) -> dict[str, str]:
    return stage_tokens(axes_theme(ax))


#: Camera-colour rule (UX-7, revised for UX2-1 in round 2): every mark of a camera (badge,
#: bar, dot, raster cell, line, frame, 3D glyph) is drawn in ONE colour per theme,
#: :func:`camera_fill`. A poster colour whose WCAG contrast against the figure background is
#: below this value is replaced on that theme by its substitute fill
#: (:data:`CAMERA_THEME_FILLS`): cameras 1 and 5 on Light and Sepia, camera 6 on Dark.
#: Only *badges* of those cameras also get a thin ring in ``stage_tokens()["outline"]``;
#: dots, bars, raster cells and lines never get an outline (round 2: outlined dots read as
#: hollow rings, outlined lines as white lines).
CAMERA_OUTLINE_CONTRAST = 2.1
#: Edge width (points) of the badge ring for substituted cameras. The other kinds stay for
#: compatibility and are no longer used for camera marks.
OUTLINE_WIDTH = {"badge": 1.2, "patch": 1.2, "line": 1.0, "dot": 0.6}


def _needs_edge(ax: Any, colour: str) -> bool:
    """True when ``colour`` would melt into the stage (contrast below
    :data:`CAMERA_OUTLINE_CONTRAST`)."""
    return contrast(colour, _tokens_for(ax)["stage"]) < CAMERA_OUTLINE_CONTRAST


def _theme_of(where: Any) -> str:
    """A theme name from an axes, a figure, a theme name or None (active theme)."""
    if where is None or isinstance(where, str):
        return resolve_theme(where)
    explicit = getattr(where, "_edu_theme", None)
    if explicit:
        return resolve_theme(explicit)
    if hasattr(where, "figure") and hasattr(where, "get_facecolor"):
        return axes_theme(where)
    return resolve_theme(None)


def camera_needs_outline(index: int, theme: Any = None) -> bool:
    """True when camera ``index``'s poster colour is too faint on ``theme`` (a theme name, an
    axes or None for the active theme), so it is drawn in its substitute fill and its
    *badge* gets the outline ring (:func:`camera_edge`)."""
    stage = stage_tokens(_theme_of(theme))["stage"]
    return contrast(camera_colour(index), stage) < CAMERA_OUTLINE_CONTRAST


def camera_edge(index: int, theme: Any = None, kind: str = "badge") -> str:
    """Edge colour for camera ``index``'s mark of ``kind``: the outline colour for a
    ``"badge"`` of a substituted camera (:func:`camera_needs_outline`), else ``"none"``
    (dots, bars, raster cells, lines and 3D glyphs never get an outline)."""
    if kind != "badge" or not camera_needs_outline(index, theme):
        return "none"
    return stage_tokens(_theme_of(theme))["outline"]


def camera_line_effects(index: int, theme: Any = None, linewidth: float = 1.5) -> list:
    """Path effects for a line in camera ``index``'s colour: always ``[]`` since round 2
    (lines use :func:`camera_fill`, which reads on every theme). Kept for compatibility."""
    del index, theme, linewidth
    return []


def camera_mark_kwargs(index: int, theme: Any = None, kind: str = "patch") -> dict[str, Any]:
    """Keyword arguments for one camera mark in :func:`camera_fill`: ``kind`` ``"patch"``
    (bars, raster cells: facecolor, no edge), ``"badge"`` (facecolor + the outline ring for
    substituted cameras), ``"dot"`` (scatter: color, no edge) or ``"line"``
    (color/linewidth). Never hollow, never an outline-coloured line (UX2-1)."""
    colour = camera_fill(index, theme)
    if kind == "dot":
        return {"color": colour, "edgecolors": "none", "linewidths": 0.0}
    if kind == "line":
        return {"color": colour, "linewidth": 1.5}
    edge = camera_edge(index, theme, "badge") if kind == "badge" else "none"
    return {"facecolor": colour, "edgecolor": edge,
            "linewidth": OUTLINE_WIDTH["badge"] if edge != "none" else 0.0}


# --------------------------------------------------------------------------------------
# Continuous colour (errors) and the shared error vocabulary

#: Dark backgrounds: viridis starts at this fraction so its darkest purple stays visible
#: (UX-6). Light themes use the full map; every value-coloured dot also gets a thin
#: ``point_edge`` (see :func:`value_dot_kwargs`).
DARK_CMAP_START = 0.18
_DARK_CMAP_NAME = "edu_viridis_dark"


def error_cmap(theme: Any = None) -> str:
    """Name of the continuous colour map for ``theme`` (registered with matplotlib, so the
    name also works for PyVista): ``"viridis"`` on Light/Sepia, viridis clipped to
    [:data:`DARK_CMAP_START`, 1] on Dark."""
    name = _theme_of(theme)
    if relative_luminance(stage_tokens(name)["stage"]) >= 0.2:
        return CONTINUOUS_CMAP
    import matplotlib
    from matplotlib.colors import ListedColormap
    if _DARK_CMAP_NAME not in matplotlib.colormaps:
        base = matplotlib.colormaps[CONTINUOUS_CMAP]
        colours = base(np.linspace(DARK_CMAP_START, 1.0, 256))
        matplotlib.colormaps.register(ListedColormap(colours, name=_DARK_CMAP_NAME))
    return _DARK_CMAP_NAME


def value_dot_kwargs(theme: Any = None, size: float = 18.0) -> dict[str, Any]:
    """scatter() kwargs for dots coloured by a value: a 0.4 pt ``point_edge`` so the
    darkest and lightest map colours never vanish into the background."""
    t = stage_tokens(_theme_of(theme))
    return {"s": size, "edgecolors": t["point_edge"], "linewidths": 0.4}


#: The one name for the error every phase reports (ED-19): the distance in the image
#: between where a corner was found and where the model puts it.
ERROR_NAME = "reprojection error"
ERROR_UNIT = "px"
_STAT_WORDS = {None: "", "mean": "mean ", "rms": "RMS ", "median": "median ",
               "max": "largest ", "p95": "95th-percentile "}
#: Fixed colour scale for reprojection error in every phase and stage (ED-19, UX-11): log,
#: 0.1 px .. 100 px. Values above the cap draw in the top colour and are listed as outliers
#: ("▲"); values below the floor draw in the bottom colour. One scale means one colour =
#: one error size from Phase 2 to Phase 4, and both a 122 px start and a 0.3 px result
#: stay readable.
ERROR_SCALE_PX = (0.1, 100.0)
ERROR_TICKS_PX = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def error_label(statistic: str | None = "mean", *, per: str | None = None,
                unit: bool = True, short: bool = False) -> str:
    """The label for reprojection error, the same words in every phase.

    ``statistic``: ``"mean"`` | ``"rms"`` | ``"median"`` | ``"max"`` | ``"p95"`` | None.
    ``per``: e.g. ``"photo"`` or ``"camera"`` (appended as "per photo"). ``short`` drops
    "reprojection" (for tight axes: "mean error (px)").
    Examples: ``error_label()`` -> "mean reprojection error (px)",
    ``error_label("rms", per="photo")`` -> "RMS reprojection error per photo (px)".
    """
    words = _STAT_WORDS[statistic.casefold() if statistic else None]
    name = "error" if short else ERROR_NAME
    text = f"{words}{name}" + (f" per {per}" if per else "")
    return f"{text} ({ERROR_UNIT})" if unit else text


class ErrorScale:
    """The shared colour scale for reprojection error (see :data:`ERROR_SCALE_PX`)."""

    def __init__(self, theme: Any = None, lo: float = ERROR_SCALE_PX[0],
                 hi: float = ERROR_SCALE_PX[1]) -> None:
        self.theme = _theme_of(theme)
        self.lo, self.hi = float(lo), float(hi)
        self.cmap = error_cmap(self.theme)
        self.label = ERROR_UNIT
        self.ticks = tuple(t for t in ERROR_TICKS_PX if self.lo <= t <= self.hi)

    @property
    def clim(self) -> tuple[float, float]:
        return self.lo, self.hi

    def norm(self):
        """A clipping LogNorm over the scale."""
        from matplotlib.colors import LogNorm
        return LogNorm(vmin=self.lo, vmax=self.hi, clip=True)

    def colours(self, values: Any) -> np.ndarray:
        """RGBA colours for ``values`` (px)."""
        import matplotlib
        vals = np.clip(np.asarray(values, dtype=float), self.lo, self.hi)
        return matplotlib.colormaps[self.cmap](self.norm()(vals))

    def over_cap(self, values: Any) -> np.ndarray:
        """Boolean mask of values above the cap (draw them as ▲ outliers)."""
        return np.asarray(values, dtype=float) > self.hi

    def mappable(self):
        """A ScalarMappable for a colour bar without data."""
        from matplotlib.cm import ScalarMappable
        mappable = ScalarMappable(norm=self.norm(), cmap=self.cmap)
        mappable._edu_quantity = "error"
        return mappable

    def scatter_kwargs(self) -> dict[str, Any]:
        """``c=values`` companions for ``ax.scatter``: cmap and norm."""
        return {"cmap": self.cmap, "norm": self.norm()}

    def log_axis(self, ax: Any, axis: str = "y") -> None:
        """A log axis with this scale's round ticks (0.1 0.3 1 3 ... px), plain numbers."""
        from matplotlib.ticker import FixedLocator, NullLocator
        target = ax.yaxis if axis == "y" else ax.xaxis
        (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
        target.set_major_locator(FixedLocator(list(self.ticks)))
        target.set_minor_locator(NullLocator())
        target.set_major_formatter(__import__("matplotlib.ticker", fromlist=["x"])
                                   .FuncFormatter(lambda v, _p: f"{v:g}"))

    def log_range(self, values: Any) -> tuple[float, float]:
        """Round log-axis limits from this scale's ticks around ``values`` (px): the
        largest tick at or below 0.95 x the smallest value and the smallest tick at or
        above 1.05 x the largest (the scale's ends when outside)."""
        vals = np.asarray(values, dtype=float).ravel()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        ticks = self.ticks or (self.lo, self.hi)
        if not vals.size:
            return float(ticks[0]), float(ticks[-1])
        vmin, vmax = float(vals.min()), float(vals.max())
        lo = max([tk for tk in ticks if tk <= vmin * 0.95] or [ticks[0]])
        hi = min([tk for tk in ticks if tk >= vmax * 1.05] or [ticks[-1]])
        if hi <= lo:
            hi = lo * 3
        return float(lo), float(hi)

    def scene_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``Scene3D.add_points``/``add_board``/``add_rays``."""
        return {"cmap": self.cmap, "clim": self.clim, "log_scale": True}

    def colourbar(self, fig: Any, ax: Any, mappable: Any = None, *, label: str | None = None,
                  shrink: float = 0.82) -> Any:
        """A slim vertical colour bar for this scale beside ``ax`` with log ticks in px."""
        from matplotlib.ticker import FixedLocator, NullLocator
        mappable = mappable if mappable is not None else self.mappable()
        mappable._edu_quantity = "error"
        bar = colourbar_px(fig, mappable, ax, label or self.label, shrink=shrink)
        bar.ax.yaxis.set_major_locator(FixedLocator(list(self.ticks)))
        bar.ax.yaxis.set_minor_locator(NullLocator())
        bar.ax.set_yticklabels([f"{t:g}" for t in self.ticks])
        return bar


def error_scale(theme: Any = None) -> ErrorScale:
    """The one colour scale for reprojection error (log 0.1-100 px), for ``theme``."""
    return ErrorScale(theme)


#: When to use which colour scale (UX2-7, round 2):
#:
#: * :func:`error_scale` (log 0.1-100 px, the default): every reprojection-error colour a
#:   learner may compare across panels, stages or phases (corners, photos, cameras, 3D
#:   points, residual arrows, solver frames). One colour = one error size everywhere.
#: * :func:`local_scale` (linear, this panel's own data range): only when the panel's point
#:   is the *pattern inside* a narrow range that the shared scale would paint as one colour
#:   (largest / smallest below about :data:`LOCAL_SCALE_MAX_RATIO`, e.g. a 1-3 px "where in
#:   the image" heat map). It always carries its own colour bar whose title says so
#:   ("px · this map only"); never use it for a panel that sits beside a shared-scale panel
#:   without that title.
LOCAL_SCALE_MAX_RATIO = 10.0
LOCAL_SCALE_NOTE = "this map only"


class LocalScale:
    """A linear colour scale over one panel's own value range (see the note above
    :data:`LOCAL_SCALE_MAX_RATIO`): theme-aware viridis, round limits, 3-5 round ticks and a
    colour bar titled ``"<unit> · this map only"``."""

    def __init__(self, values: Any, theme: Any = None, *, lo: float | None = 0.0,
                 hi: float | None = None, unit: str = ERROR_UNIT,
                 note: str = LOCAL_SCALE_NOTE, cmap: str | None = None) -> None:
        from matplotlib.ticker import MaxNLocator
        vals = np.asarray(values, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        self.theme = _theme_of(theme)
        # UX3-3: viridis only for errors / displacements in px, µm or mm; any other unit
        # (points, photos, share, ...) gets the one-hue amount map.
        self.cmap = cmap or (error_cmap(self.theme) if is_error_unit(unit)
                             else amount_cmap(self.theme))
        self.quantity = "error" if self.cmap == error_cmap(self.theme) else "amount"
        data_lo = float(vals.min()) if vals.size else 0.0
        data_hi = float(vals.max()) if vals.size else 1.0
        low = data_lo if lo is None else float(lo)
        high = data_hi if hi is None else float(hi)
        if high <= low:
            high = low + (abs(low) * 0.1 or 1.0)
        ticks = [float(t) for t in MaxNLocator(4, steps=[1, 2, 2.5, 5, 10])
                 .tick_values(low, high)]
        # Round the top up to the next tick so the bar ends on a labelled value.
        above = [t for t in ticks if t >= high - 1e-12]
        self.lo, self.hi = low, (above[0] if above else high)
        self.ticks = tuple(t for t in ticks if self.lo - 1e-9 <= t <= self.hi + 1e-9)
        self.unit = unit
        self.note = note
        self.label = f"{unit} · {note}" if note else unit
        self.data_range = (data_lo, data_hi)

    @property
    def clim(self) -> tuple[float, float]:
        return self.lo, self.hi

    def norm(self):
        from matplotlib.colors import Normalize
        return Normalize(vmin=self.lo, vmax=self.hi, clip=True)

    def colours(self, values: Any) -> np.ndarray:
        import matplotlib
        return matplotlib.colormaps[self.cmap](self.norm()(np.asarray(values, dtype=float)))

    def mappable(self):
        from matplotlib.cm import ScalarMappable
        mappable = ScalarMappable(norm=self.norm(), cmap=self.cmap)
        mappable._edu_quantity = self.quantity
        return mappable

    def scatter_kwargs(self) -> dict[str, Any]:
        return {"cmap": self.cmap, "norm": self.norm()}

    def imshow_kwargs(self) -> dict[str, Any]:
        """``ax.imshow(grid, **scale.imshow_kwargs())`` for a heat map."""
        return {"cmap": self.cmap, "norm": self.norm(), "interpolation": "nearest"}

    def scene_kwargs(self) -> dict[str, Any]:
        return {"cmap": self.cmap, "clim": self.clim, "log_scale": False}

    def tick_text(self, value: float) -> str:
        step = min(np.diff(self.ticks)) if len(self.ticks) > 1 else abs(self.hi)
        decimals = 0 if step >= 1 else (1 if step >= 0.1 else 2)
        return f"{value:.{decimals}f}"

    def colourbar(self, fig: Any, ax: Any, mappable: Any = None, *, label: str | None = None,
                  shrink: float = 0.82) -> Any:
        """A slim vertical colour bar beside ``ax`` with this panel's round ticks."""
        from matplotlib.ticker import FixedLocator, NullLocator
        mappable = mappable if mappable is not None else self.mappable()
        mappable._edu_quantity = self.quantity
        bar = colourbar_px(fig, mappable, ax, self.unit, shrink=shrink)
        bar.ax.yaxis.set_major_locator(FixedLocator(list(self.ticks)))
        bar.ax.yaxis.set_minor_locator(NullLocator())
        bar.ax.set_yticklabels([self.tick_text(t) for t in self.ticks])
        note = label if label is not None else self.note
        if note:  # "this map only", written along the bar so a narrow panel keeps it
            bar.ax.set_ylabel(note, rotation=270, labelpad=11, fontsize=TICK_SIZE,
                              color=_tokens_for(ax)["muted"])
        return bar


def local_scale(values: Any, theme: Any = None, *, lo: float | None = 0.0,
                hi: float | None = None, unit: str = ERROR_UNIT,
                note: str = LOCAL_SCALE_NOTE) -> LocalScale:
    """A linear colour scale over ``values``' own range (default 0 .. a round value at or
    above the largest), for maps whose pattern the shared log scale would wash out. ``lo=None``
    starts at the smallest value. See the note above :data:`LOCAL_SCALE_MAX_RATIO`."""
    return LocalScale(values, theme, lo=lo, hi=hi, unit=unit, note=note)


def needs_local_scale(values: Any) -> bool:
    """True when ``values`` (> 0) span less than :data:`LOCAL_SCALE_MAX_RATIO` (largest /
    smallest), i.e. the shared log error scale would draw them as nearly one colour."""
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 2:
        return False
    return float(vals.max() / vals.min()) < LOCAL_SCALE_MAX_RATIO


# --------------------------------------------------------------------------------------
# Which colour map for which quantity (UX3-3, round 3)
#
# One rule for all four phases, so "yellow = far off" means the same in every panel:
#
# * :func:`error_cmap` (viridis, yellow = large) ONLY for reprojection error and
#   displacements measured in px, µm or mm (:class:`ErrorScale`, :class:`LocalScale` with an
#   error unit, error-coloured 3D points). Never for counts or shares.
# * :func:`amount_cmap` (one hue: the stage colour -> the theme's amount hue, which is the
#   accent unless that looks like viridis) for counts, density, coupling, shares, scores:
#   any "how much" that is not an error (:func:`amount_scale`).
# * :func:`signed_cmap` (diverging blue <- neutral -> magenta, symmetric about 0) for signed
#   quantities: grew vs shrank, +/- change (:func:`signed_scale`).
# * Label every colour bar with its unit ("points", "share", "photos"), so it never reads as
#   a px bar. :func:`colour_map_misuse` finds the error map on non-error data where it can
#   tell (tests, review screenshots).

#: Units that may use the error map (errors and displacements).
ERROR_MAP_UNITS = ("px", "µm", "um", "mm", "pixel", "pixels")
#: An amount hue closer than this (CIE76) to any viridis colour is replaced (Sepia's teal
#: accent is ~10 from viridis' middle).
AMOUNT_MIN_VIRIDIS_DISTANCE = 25.0
_AMOUNT_HUE_FALLBACKS = ("#1D4ED8", "#6D28D9", "#93A0FF")
#: Diverging ends per light/dark background: (negative, positive).
_SIGNED_ENDS = {"light": ("#1D4ED8", "#A21CAF"), "dark": ("#60A5FA", "#E879F9")}
_VIRIDIS_SAMPLES: list[str] = []


def is_error_unit(unit: Any) -> bool:
    """True when ``unit`` (a colour-bar unit such as "px" or "µm · this map only") is one the
    error map may show (:data:`ERROR_MAP_UNITS`)."""
    import re
    text = str(unit or "").lower().replace("μ", "µ")   # Greek mu -> micro sign
    words = re.findall(r"[a-zµ]+", text)
    return any(w in ERROR_MAP_UNITS for w in words)


def viridis_distance(colour: str) -> float:
    """Smallest CIE76 distance from ``colour`` to the viridis map (64 samples)."""
    if not _VIRIDIS_SAMPLES:
        import matplotlib
        from matplotlib.colors import to_hex
        _VIRIDIS_SAMPLES.extend(to_hex(c) for c in
                                matplotlib.colormaps[CONTINUOUS_CMAP](np.linspace(0, 1, 64)))
    return min(colour_distance(colour, v) for v in _VIRIDIS_SAMPLES)


def amount_hue(theme: Any = None) -> str:
    """The top colour of :func:`amount_cmap` on ``theme``: the figure accent unless it is
    within :data:`AMOUNT_MIN_VIRIDIS_DISTANCE` of viridis or lacks contrast on the stage
    (then a fixed blue / violet)."""
    t = stage_tokens(_theme_of(theme))
    for colour in (t["accent"],) + _AMOUNT_HUE_FALLBACKS:
        if viridis_distance(colour) >= AMOUNT_MIN_VIRIDIS_DISTANCE and \
                contrast(colour, t["stage"]) >= MIN_STAGE_CONTRAST:
            return colour
    return _AMOUNT_HUE_FALLBACKS[0]


def _register_cmap(name: str, colours: Sequence[Any]) -> str:
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    if name not in matplotlib.colormaps:
        matplotlib.colormaps.register(LinearSegmentedColormap.from_list(name, list(colours)))
    return name


def _mix(a: str, b: str, share_b: float) -> tuple[float, float, float]:
    ca, cb = np.array(_rgb(a)), np.array(_rgb(b))
    return tuple(float(v) for v in ca * (1 - share_b) + cb * share_b)


def amount_cmap(theme: Any = None) -> str:
    """Name of the one-hue sequential map for counts, density, coupling and shares (UX3-3):
    from a 10 % tint of :func:`amount_hue` on the stage colour to the hue itself. Registered
    with matplotlib (the name also works for PyVista). Never viridis."""
    name = _theme_of(theme)
    t = stage_tokens(name)
    hue = amount_hue(name)
    return _register_cmap(f"edu_amount_{name.casefold()}", [_mix(t["stage"], hue, 0.1), hue])


def signed_cmap(theme: Any = None) -> str:
    """Name of the diverging map for signed quantities (UX3-3): blue (negative) <- a neutral
    near the stage colour (0) -> magenta (positive). Use with a range symmetric about 0
    (:func:`signed_scale`)."""
    name = _theme_of(theme)
    t = stage_tokens(name)
    dark = relative_luminance(t["stage"]) < 0.2
    neg, pos = _SIGNED_ENDS["dark" if dark else "light"]
    return _register_cmap(f"edu_signed_{name.casefold()}",
                          [neg, _mix(t["stage"], t["muted"], 0.18), pos])


def is_error_cmap(cmap: Any) -> bool:
    """True when ``cmap`` (a name or a Colormap) is the error map (viridis or its Dark cut)."""
    name = getattr(cmap, "name", cmap)
    return str(name) in (CONTINUOUS_CMAP, _DARK_CMAP_NAME)


def amount_scale(values: Any, theme: Any = None, *, unit: str = "count",
                 lo: float | None = 0.0, hi: float | None = None) -> LocalScale:
    """A linear colour scale for a non-error amount (counts, density, coupling, share) in
    :func:`amount_cmap`, with round ticks and a colour bar titled ``unit``. For a log
    density pass ``norm=matplotlib.colors.LogNorm(...)`` with ``cmap=scale.cmap``."""
    scale = LocalScale(values, theme, lo=lo, hi=hi, unit=unit, note="",
                       cmap=amount_cmap(theme))
    scale.quantity = "amount"
    return scale


def signed_scale(values: Any, theme: Any = None, *, unit: str = "") -> LocalScale:
    """A linear colour scale symmetric about 0 in :func:`signed_cmap` for signed values."""
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    top = float(np.abs(vals).max()) if vals.size else 1.0
    top = _nice_ceiling(top) if top > 0 else 1.0
    scale = LocalScale(values, theme, lo=-top, hi=top, unit=unit, note="",
                       cmap=signed_cmap(theme))
    scale.lo, scale.hi = -top, top
    scale.quantity = "signed"
    return scale


# --------------------------------------------------------------------------------------
# Limits and cursors (UX3-11): red only when something failed

#: Dash pattern of a limit / reference line.
LIMIT_DASHES = (0, (4, 3))


def limit_style(where: Any = None, crossed: bool = False) -> dict[str, Any]:
    """Line kwargs for a limit or threshold: muted grey dashes while no value crosses it
    (the 'limit not crossed' neutral style), the theme's ``bad`` red only when one does."""
    t = stage_tokens(_theme_of(where))
    return {"color": t["bad"] if crossed else t["muted"], "linestyle": LIMIT_DASHES,
            "linewidth": 1.3 if crossed else 1.1, "zorder": 2.5}


def limit_crossed(values: Any, limit: float, *, above_is_bad: bool = True) -> bool:
    """True when any finite value is beyond ``limit`` (above it, or below with
    ``above_is_bad=False``)."""
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return False
    return bool(np.any(vals > limit) if above_is_bad else np.any(vals < limit))


def limit_line(ax: Any, value: float, *, values: Any = None, crossed: bool | None = None,
               axis: str = "y", above_is_bad: bool = True, label: str | None = None) -> Any:
    """A limit / threshold line on ``ax`` (``axis="y"``: horizontal at ``value``; ``"x"``:
    vertical) in :func:`limit_style`: neutral dashes unless ``crossed`` (or, given
    ``values``, unless one of them is beyond the limit). ``label`` is written at the line's
    end in the same colour. Returns the line (``line._edu_limit_crossed``)."""
    if crossed is None:
        crossed = limit_crossed(values, value, above_is_bad=above_is_bad) \
            if values is not None else False
    kw = limit_style(ax, bool(crossed))
    line = ax.axhline(value, **kw) if axis == "y" else ax.axvline(value, **kw)
    line._edu_limit_crossed = bool(crossed)
    if label:
        from matplotlib.transforms import blended_transform_factory
        if axis == "y":
            where = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(0.995, value, f" {label}", transform=where, ha="right", va="bottom",
                    fontsize=TICK_SIZE, color=kw["color"])
        else:
            where = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(value, 0.99, f" {label}", transform=where, ha="left", va="top",
                    fontsize=TICK_SIZE, color=kw["color"])
    return line


def cursor_style(where: Any = None) -> dict[str, Any]:
    """Line kwargs for a playback cursor (the current step): the accent, never red."""
    return {"color": stage_tokens(_theme_of(where))["accent"], "linewidth": 1.6, "zorder": 4}


# --------------------------------------------------------------------------------------
# Text that must stay inside its card (UX3-1)

#: Smallest font a card text may shrink to before the layout wraps no further.
CARD_TEXT_MIN_SIZE = 7.0


def card_text(ax: Any, x: float, y: float, text: str, *, card: Any, max_lines: int = 2,
              min_size: float = CARD_TEXT_MIN_SIZE, pad_px: float = 6.0, **kwargs: Any) -> Any:
    """``ax.text`` that the layout keeps inside ``card`` (a patch, e.g. a FancyBboxPatch):
    after every layout it is wrapped at the measured width (up to ``max_lines`` lines) and
    shrunk towards ``min_size`` until it fits with ``pad_px`` to the card's edge. Use for
    every finding written on a card (gate cards, legends drawn as cards)."""
    label = ax.text(x, y, text, **kwargs)
    label._edu_card = card
    label._edu_card_full = str(text)
    label._edu_card_size = float(label.get_fontsize())
    label._edu_card_lines = max(1, int(max_lines))
    label._edu_card_min = float(min_size)
    label._edu_card_pad = float(pad_px)
    return label


def _wrap_to_width(label: Any, words: list[str], limit: float, renderer: Any) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        label.set_text(trial)
        if current and label.get_window_extent(renderer).width > limit:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _fit_card_texts(fig: Any, renderer: Any) -> None:
    for ax in fig.axes:
        for label in ax.texts:
            card = getattr(label, "_edu_card", None)
            if card is None or not label.get_visible():
                continue
            try:
                box = card.get_window_extent(renderer)
                full = label._edu_card_full
                pad = label._edu_card_pad * fig.dpi / 100.0
                label.set_fontsize(label._edu_card_size)
                label.set_text(full)
                ext = label.get_window_extent(renderer)
                ha = label.get_horizontalalignment()
                if ha == "left":
                    limit = box.x1 - pad - ext.x0
                elif ha == "right":
                    limit = ext.x1 - (box.x0 + pad)
                else:
                    limit = box.width - 2 * pad
                words = full.split()
                size = label._edu_card_size
                best = None
                while True:
                    label.set_fontsize(size)
                    lines = _wrap_to_width(label, words, limit, renderer)
                    if len(lines) <= label._edu_card_lines:
                        label.set_text("\n".join(lines))
                        if label.get_window_extent(renderer).width <= limit + 0.5:
                            best = (size, lines)
                            break
                    if size <= label._edu_card_min:
                        best = (size, lines)
                        break
                    size = max(label._edu_card_min, size - 0.5)
                label.set_fontsize(best[0])
                label.set_text("\n".join(best[1]))
            except Exception:  # noqa: BLE001 - a card text never breaks the drawing
                continue


# --------------------------------------------------------------------------------------
# Numbers, photos and cameras as the learner reads them

#: Photos are numbered from 1 everywhere a learner reads them (SCI-8), like cameras:
#: stored index 0 is "Photo 1". Numbers drawers may add the stored 0-based index in a
#: column titled "index (0-based)".
PHOTO_NUMBER_BASE = 1


def photo_number(index: int) -> int:
    """The number a learner sees for the stored (0-based) photo ``index``."""
    return int(index) + PHOTO_NUMBER_BASE


def photo_label(index: int, *, short: bool = False) -> str:
    """``"Photo 17"`` for stored index 16 (``short``: ``"17"``)."""
    return str(photo_number(index)) if short else f"Photo {photo_number(index)}"


def photos_label(indices: Sequence[int], *, limit: int = 8) -> str:
    """``"Photos 3, 7–9 and 12"`` for stored indices; runs of 3+ become ranges; more than
    ``limit`` groups end with ``"+N more"``."""
    numbers = sorted({photo_number(i) for i in indices})
    if not numbers:
        return "no photos"
    groups: list[str] = []
    start = prev = numbers[0]
    for n in numbers[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        groups.append(f"{start}–{prev}" if prev - start >= 2 else
                      (f"{start}, {prev}" if prev != start else f"{start}"))
        if n is not None:
            start = prev = n
    flat = ", ".join(groups).split(", ")
    if len(flat) > limit:
        flat = flat[:limit] + [f"+{len(flat) - limit} more"]
    word = "Photo" if len(numbers) == 1 else "Photos"
    body = flat[0] if len(flat) == 1 else ", ".join(flat[:-1]) + " and " + flat[-1]
    return f"{word} {body}"


def camera_title(index: int, name: str | None = None, *, max_name: int = 22) -> str:
    """Camera picker / legend text: ``"Camera 4 · cam4_FrontLeft"`` (the name is dropped
    when it only repeats the number, and elided after ``max_name`` characters)."""
    number = camera_badge_label(index)
    text = f"Camera {number}"
    if name:
        plain = str(name).strip()
        folded = plain.casefold().replace(" ", "").replace("_", "")
        if folded not in {number, f"cam{number}", f"camera{number}", f"c{number}"}:
            if len(plain) > max_name:
                plain = plain[:max_name - 1] + "…"
            text += f" · {plain}"
    return text


def format_number(value: float, decimals: int | None = None) -> str:
    """One number formatting rule: < 10 -> 2 decimals, < 100 -> 1, else 0 with thousands
    separators (``decimals`` overrides). ``0.1`` -> "0.10", ``3.2954`` -> "3.30",
    ``121.9`` -> "122", ``15447`` -> "15,447"."""
    v = float(value)
    if not np.isfinite(v):
        return "–"
    if decimals is None:
        a = abs(v)
        decimals = 2 if a < 10 else (1 if a < 100 else 0)
    return f"{v:,.{decimals}f}"


def format_sig(value: float, digits: int = 2) -> str:
    """``value`` to ``digits`` significant digits, never in exponent form and with thousands
    separators: ``0.0834`` -> "0.083", ``0.14`` -> "0.14", ``6.81`` -> "6.8", ``52.7`` -> "53",
    ``139`` -> "140" (UX2-14: one precision for a row of parameter labels)."""
    v = float(value)
    if not np.isfinite(v):
        return "–"
    if v == 0:
        return "0"
    magnitude = int(np.floor(np.log10(abs(v))))
    decimals = max(digits - 1 - magnitude, 0)
    rounded = round(v, digits - 1 - magnitude)
    return f"{rounded:,.{decimals}f}"


def format_px(value: float, decimals: int | None = None) -> str:
    """``"0.30 px"`` (see :func:`format_number`)."""
    return f"{format_number(value, decimals)} px"


def format_um(value_mm: float) -> str:
    """A length given in mm, as µm below 1 mm (``0.172`` -> "172 µm") else mm."""
    v = float(value_mm)
    return f"{format_number(v * 1000, 0)} µm" if abs(v) < 1 else f"{format_number(v)} mm"


def format_count(count: int, noun: str | None = None, plural: str | None = None) -> str:
    """``"15,447"`` or with a noun ``"1 photo"`` / ``"5 photos"``."""
    n = int(count)
    text = f"{n:,}"
    if noun:
        text += " " + (noun if n == 1 else (plural or noun + "s"))
    return text


def format_percent(fraction: float, decimals: int = 0) -> str:
    """``0.774`` -> ``"77 %"``."""
    return f"{100 * float(fraction):.{decimals}f} %"


def format_change(before: float, after: float, unit: str = "px") -> str:
    """``"3.30 → 0.30 px"``."""
    return f"{format_number(before)} → {format_number(after)}" + (f" {unit}" if unit else "")


# --------------------------------------------------------------------------------------
# Glossary (ED-18): plain words for the terms a stage cannot avoid

GLOSSARY: dict[str, str] = {
    "reprojection error": "How far, in pixels, a corner the detector found is from where the "
                          "fitted camera model says it should appear. Smaller is better.",
    "RMS": "Root mean square: square every error, average, take the square root. It weighs "
           "large errors more than the plain mean does.",
    "mean error": "The average reprojection error over all detected corners.",
    "residual": "The leftover difference between a measurement and the model's prediction.",
    "pose": "Where something is and which way it faces (a position and a rotation).",
    "intrinsics": "A camera's own settings: magnification or focal length, image centre "
                  "and lens distortion.",
    "extrinsics": "Where a camera sits and which way it points in the rig.",
    "principal point": "The pixel where the lens's optical axis meets the sensor.",
    "telecentric": "A lens that sees with parallel rays: objects look the same size at any "
                   "distance, so depth cannot be read from one photo.",
    "pinhole": "The usual camera model: rays meet in one point, so far things look smaller.",
    "distortion": "Bending of straight lines by the lens, worst near the image edges.",
    "board view": "One flat face of the target seen in one photo, used as a flat target "
                  "of its own.",
    "homography": "The perspective warp that maps a flat printed pattern onto its photo.",
    "bundle adjustment": "Fitting all cameras and all target poses together so every ray "
                         "meets its corner.",
    "cost": "The solver's score: half the sum of squared errors (px²). It falls as the fit "
            "improves.",
    "Jacobian": "A table of how much each error changes when each unknown is nudged.",
    "Schur complement": "A shortcut: solve the small camera system first, then fill in each "
                        "photo's pose on its own.",
    "gauge freedom": "A symmetry of the photos: moving, turning or scaling the whole rig and "
                     "target together leaves every photo the same. For a telecentric lens, "
                     "scaling changes its magnification and telecentricity instead, and a sideways "
                     "shift is only nearly a symmetry. The solver removes these by fixing one "
                     "photo's pose and the printed positions of its reference corners, so it "
                     "solves for one specific form.",
    "null space": "Directions in the unknowns that the photos do not fix: moving along one "
                  "leaves every error unchanged. Symmetries are not counted here, since the "
                  "solver fixes them first. What remains is a corner detected by one camera in "
                  "one photo, which may slide along its ray, or other weakly fixed lens or pose "
                  "directions.",
    "adaptive threshold": "Each pixel is compared with the average of its neighbourhood, so "
                          "dim and bright parts of a photo both work.",
    "sub-pixel": "Finer than one pixel: the corner is placed between pixel centres.",
    "seed": "The first guess the solver starts from.",
    "outlier": "A value far from the rest, often a bad detection.",
}


def glossary(term: str) -> str:
    """Plain-words definition of ``term`` (case-insensitive), or "" when unknown."""
    key = str(term).strip().casefold()
    for name, text in GLOSSARY.items():
        if name.casefold() == key:
            return text
    return ""


def glossary_terms_in(text: str) -> list[str]:
    """Glossary terms that occur in ``text`` (whole words, case-insensitive), in order."""
    import re
    found = []
    for name in GLOSSARY:
        match = re.search(r"\b" + re.escape(name) + r"\b", text, flags=re.IGNORECASE)
        if match:
            found.append((match.start(), name))
    return [name for _, name in sorted(found)]


# --------------------------------------------------------------------------------------
# Caption limits (layout contract)

#: Most sentences a caption may have per detail level (brief: 2 / 3 / 4).
CAPTION_MAX_SENTENCES = {1: 2, 2: 3, 3: 4}


def count_sentences(text: str) -> int:
    """Sentences in ``text`` (ends at . ! ? followed by a space or the end; "e.g." and
    decimals do not end a sentence)."""
    import re
    plain = re.sub(r"\b(e\.g|i\.e|etc|vs|approx|cf|Fig|ca)\.", "x", str(text).strip())
    plain = re.sub(r"(\d)\.(\d)", r"\1\2", plain)
    parts = [p for p in re.split(r"[.!?…]+(?:\s+|$)", plain) if p.strip()]
    return len(parts)


#: Most *rendered lines* a caption may take per detail level, measured at the narrowest
#: supported dialog size :data:`NARROWEST_DIALOG` (UX2-5, round 2: a 60-word sentence
#: joined with semicolons passes a sentence count but not this). Same numbers as the
#: brief's sentence limits; the stage reserves exactly this many lines.
CAPTION_MAX_LINES = {1: 2, 2: 3, 3: 4}
#: The narrowest supported dialog size (width, height) in pixels.
NARROWEST_DIALOG = (1280, 800)
#: Dialog widths at which every caption must keep its line limit and have no one-word last
#: line (UX3-2, round 3: the line check ran at 1280 only, while 1400 left widows).
SUPPORTED_DIALOG_WIDTHS = (1280, 1400)


def caption_problems(text: str, level: int, lines: int | None = None, *,
                     lines_by_width: Mapping[int, int] | None = None,
                     last_lines: Mapping[int, str] | None = None) -> list[str]:
    """Contract problems of a caption at ``level``: too many sentences; when ``lines``
    (rendered lines at :data:`NARROWEST_DIALOG`) or ``lines_by_width`` ({dialog width:
    rendered lines}, see ``EducationStageView``) are given, too many lines at any width; and
    when ``last_lines`` ({dialog width: text of the last rendered line}) is given, a one-word
    last line (an orphan such as "focus).") on a caption of two or more lines."""
    level = int(level)
    limit = CAPTION_MAX_SENTENCES.get(level, 4)
    n = count_sentences(text)
    problems = [f"{n} sentences at L{level} (limit {limit})"] if n > limit else []
    line_limit = CAPTION_MAX_LINES.get(level, 4)
    counts: dict[int, int] = {}
    if lines is not None:
        counts[int(NARROWEST_DIALOG[0])] = int(lines)
    for width, count in (lines_by_width or {}).items():
        if count is not None:
            counts[int(width)] = int(count)
    for width in sorted(counts):
        if counts[width] > line_limit:
            problems.append(f"{counts[width]} lines at {width} px at L{level} "
                            f"(limit {line_limit})")
    for width in sorted(last_lines or {}):
        last = str((last_lines or {})[width] or "").strip()
        if len(last.split()) == 1 and counts.get(int(width), 2) > 1:
            problems.append(f"one-word last line \"{last}\" at {int(width)} px")
    return problems


def apply_figure_style(figure: Any, theme_name: str | None) -> None:
    """Restyle an *existing* figure's chrome for ``theme_name`` (data colours are untouched).

    Prefer redrawing inside :func:`figure_style`; use this for figures built elsewhere.
    """
    t = stage_tokens(theme_name)
    figure._edu_theme = t["theme"]
    figure.patch.set_facecolor(t["stage"])
    for ax in figure.axes:
        ax.set_facecolor(t["stage"])
        for text in ax.texts:
            gid = text.get_gid() or ""
            if gid == "edu:subtitle":
                text.set_color(t["muted"])
        if (ax.title.get_gid() or "") == "edu:title" or ax.title.get_text():
            ax.title.set_color(t["title"])
        left_title = getattr(ax, "_left_title", None)
        if left_title is not None:
            left_title.set_color(t["title"])
        ax.xaxis.label.set_color(t["text"])
        ax.yaxis.label.set_color(t["text"])
        ax.tick_params(colors=t["muted"], labelcolor=t["text"])
        for spine in ax.spines.values():
            spine.set_color(t["spine"])


# --------------------------------------------------------------------------------------
# Titles


#: Between the panel letter and the title (UX-31: one rule everywhere; never hand-write
#: "A The solver", always call :func:`panel_title`).
LETTER_GAP = "  "


def panel_title(ax: Any, letter: str | None, title: str, subtitle: str | None = None,
                *, loc: str = "left") -> None:
    """Poster title above ``ax``: bold ``"A  Camera image"`` and a muted subtitle line below it.

    Both lines live in the axes-title slot, so constrained layout reserves their space and
    they never overlap data. ``loc`` is ``"left"`` or ``"center"``. With the education
    layout engine (:func:`align_panel_titles`, on by default in ``EducationStageView``)
    titles in one row share one baseline and too-long titles/subtitles end in "…".
    """
    t = _tokens_for(ax)
    head = f"{letter}{LETTER_GAP}{title}" if letter else title
    for slot in ("left", "center", "right"):
        ax.set_title("", loc=slot)
    # The empty second line reserves room for the subtitle (baseline = last line).
    text = f"{head}\n " if subtitle else head
    title_artist = ax.set_title(text, loc=loc, fontsize=TITLE_SIZE, fontweight="bold",
                                color=t["title"], linespacing=1.35)
    title_artist.set_gid("edu:title")
    title_artist._edu_head = head
    title_artist._edu_has_subtitle = bool(subtitle)
    for old in [x for x in ax.texts if (x.get_gid() or "") == "edu:subtitle"]:
        old.remove()
    if subtitle:
        x = 0.0 if loc == "left" else 0.5
        # Axes3D.text(x, y, z, s) reads a third positional as z: use the 2D text there.
        add_text = ax.text2D if getattr(ax, "name", "") == "3d" else ax.text
        sub = add_text(x, 1.0, subtitle, transform=title_artist.get_transform(),
                       ha="left" if loc == "left" else "center", va="baseline",
                       fontsize=SUBTITLE_SIZE, color=t["muted"], gid="edu:subtitle",
                       clip_on=False, in_layout=False)
        sub._edu_full = subtitle


def panel_title_3d(ax: Any, letter: str | None, title: str, subtitle: str | None = None,
                   *, loc: str = "left") -> None:
    """:func:`panel_title` for a 3D axes (kept as an explicit name; ``panel_title`` itself
    now handles 3D axes, whose ``text`` takes ``x, y, z, s``)."""
    panel_title(ax, letter, title, subtitle, loc=loc)


# --------------------------------------------------------------------------------------
# Camera badges


def camera_label_colour(ax: Any, index: int) -> str:
    """Camera colour for *text* on ``ax``: the camera colour, or the theme text colour when
    the camera colour would be unreadable on this background (contrast < 2.2)."""
    colour = camera_fill(index, ax)
    return colour if contrast(colour, _tokens_for(ax)["stage"]) >= 2.2 else _tokens_for(ax)["text"]


def _badge_kwargs(ax: Any, index: int, size: float, highlight: bool, flagged: bool) -> dict:
    t = _tokens_for(ax)
    outline = camera_needs_outline(index, ax)
    edge = t["bad"] if flagged else (t["badge_edge"] if highlight else
                                     (t["outline"] if outline else "none"))
    width = 2.2 if (flagged or highlight) else (OUTLINE_WIDTH["badge"] if outline else 0.0)
    return dict(boxstyle=f"circle,pad={0.28 * size / 11:.3f}", facecolor=camera_fill(index, ax),
                edgecolor=edge, linewidth=width)


def layout_badges(points: Any, radius: float | Sequence[float],
                  bounds: tuple[float, float, float, float], *,
                  centre: Sequence[float] | None = None,
                  label_widths: Sequence[float] | None = None,
                  label_sides: Sequence[int] | None = None, gap: float = 3.0,
                  iterations: int = 60, square: bool = False) -> np.ndarray:
    """Screen positions for badges: start at ``points`` (N x 2, pixels), push overlapping
    badges apart (away from ``centre`` when they coincide) and keep every badge, with its
    label, inside ``bounds`` = (x0, y0, x1, y1). ``label_sides``: +1 (default) puts a
    badge's label to its right, -1 to its left. ``square``: keep the badges' bounding
    squares apart (what the text checker measures), not only their circles. Pure numpy;
    used by
    :func:`draw_camera_badges` (``repel=True``), the PyVista overlay and the static 3D
    fallback alike."""
    p = np.array(points, dtype=float).reshape(-1, 2)
    n = len(p)
    if n == 0:
        return p
    r = np.broadcast_to(np.asarray(radius, dtype=float), (n,)).copy()
    widths = np.zeros(n) if label_widths is None else np.asarray(label_widths, dtype=float)
    sides = np.ones(n) if label_sides is None else np.sign(np.asarray(label_sides, float))
    sides[sides == 0] = 1.0
    right_w = np.where(sides > 0, widths, 0.0)
    left_w = np.where(sides < 0, widths, 0.0)
    x0, y0, x1, y1 = (float(v) for v in bounds)
    mid = np.array(centre, dtype=float) if centre is not None else np.array(
        [(x0 + x1) / 2, (y0 + y1) / 2])

    def clamp(q: np.ndarray) -> np.ndarray:
        lo_x, hi_x = x0 + r + left_w + 2, x1 - r - right_w - 2
        lo_y, hi_y = y0 + r + 2, y1 - r - 2
        q[:, 0] = np.minimum(np.maximum(q[:, 0], lo_x), np.maximum(hi_x, lo_x))
        q[:, 1] = np.minimum(np.maximum(q[:, 1], lo_y), np.maximum(hi_y, lo_y))
        return q

    p = clamp(p)
    for _ in range(int(iterations)):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = p[j] - p[i]
                need = r[i] + r[j] + gap
                if square:
                    # Badge squares with their labels as rectangles; separate them along
                    # the axis they overlap least on.
                    ox = (min(p[i, 0] + r[i] + right_w[i], p[j, 0] + r[j] + right_w[j])
                          - max(p[i, 0] - r[i] - left_w[i], p[j, 0] - r[j] - left_w[j]) + gap)
                    oy = (min(p[i, 1] + r[i], p[j, 1] + r[j])
                          - max(p[i, 1] - r[i], p[j, 1] - r[j]) + gap)
                    if ox <= 0 or oy <= 0:
                        continue
                    k = 0 if ox <= oy else 1
                    ci = p[i, k] + ((right_w[i] - left_w[i]) / 2 if k == 0 else 0.0)
                    cj = p[j, k] + ((right_w[j] - left_w[j]) / 2 if k == 0 else 0.0)
                    if abs(cj - ci) < 1e-6:
                        sign = 1.0 if j % 2 else -1.0
                    else:
                        sign = float(np.sign(cj - ci))
                    push = (ox if k == 0 else oy) / 2 + 0.25
                    p[i, k] -= push * sign
                    p[j, k] += push * sign
                    moved = True
                    continue
                dist = float(np.hypot(*d))
                if dist >= need:
                    continue
                if dist < 1e-6:
                    out = p[j] - mid
                    norm = float(np.hypot(*out))
                    d = out / norm if norm > 1e-6 else np.array([1.0, 0.0])
                    if j % 2:
                        d = -d
                else:
                    d = d / dist
                push = (need - dist) / 2 + 0.25
                p[i] -= push * d
                p[j] += push * d
                moved = True
        p = clamp(p)
        if not moved:
            break
    return p


def _badge_radius_pt(size: float) -> float:
    """Approximate radius (points) of a one/two-digit badge circle at font ``size``."""
    return 0.6 * size + 0.28 * size * size / 11 + 0.5


def draw_camera_badges(ax: Any, xy: Any, indices: Sequence[int],
                       labels: Sequence[str] | None = None, *, size: float = 11,
                       highlight: Sequence[int] = (), flagged: Sequence[int] = (),
                       transform: Any = None, label_offset: tuple[float, float] = (14, 0),
                       zorder: float = 6, repel: bool = False, leaders: bool = True,
                       gap: float = 2.0) -> list:
    """Numbered circular badges in camera colours at ``xy`` (N x 2, data coords by default).

    ``labels`` (optional, e.g. ``"(0.47)"``) are drawn beside each badge, offset by
    ``label_offset`` points. ``highlight`` cameras get an outline ring, ``flagged`` a red
    ring (e.g. would be dropped). Returns the created text artists.

    ``repel=True``: badges that would overlap (each other or their labels) are pushed apart
    (``gap`` points between circles) and kept inside the axes (:func:`layout_badges`); a
    thin muted leader line joins each moved badge to its true point (``leaders``). The
    spread is computed again on every draw of an :class:`EduLayoutEngine` figure (and by
    :func:`text_collisions`), so it holds at the final panel size. The leader lines
    (gid ``"edu:leader"``) are returned after the texts.
    """
    points = np.atleast_2d(np.asarray(xy, dtype=float))
    indices = [int(i) for i in indices]
    if len(points) != len(indices):
        raise ValueError("xy and indices must have the same length")
    t = _tokens_for(ax)
    transform = ax.transData if transform is None else transform
    dx, dy = label_offset
    placed = points.copy()
    spread = bool(repel) and len(points) > 1
    if spread:
        placed = _repel_badges(ax, points, transform, [size] * len(points),
                               _estimated_label_widths(labels, size), dx, gap)
    texts, badges, extras, lines = [], [], [], []
    for k, ((x, y), cam) in enumerate(zip(placed, indices)):
        badge = ax.text(
            x, y, camera_badge_label(cam), transform=transform, ha="center", va="center",
            fontsize=size, fontweight="bold", color=badge_text_colour(cam, ax), zorder=zorder,
            bbox=_badge_kwargs(ax, cam, size, cam in highlight, cam in flagged),
            clip_on=False)
        texts.append(badge)
        badges.append(badge)
        extra = None
        if labels is not None and labels[k]:
            extra = ax.annotate(
                labels[k], (x, y), xycoords=transform, xytext=(dx, dy),
                textcoords="offset points", ha="left" if dx >= 0 else "right", va="center",
                fontsize=size * 0.9, color=t["text"], zorder=zorder, annotation_clip=False)
            texts.append(extra)
        extras.append(extra)
        if spread and leaders:
            from matplotlib.lines import Line2D
            line = Line2D([points[k, 0], x], [points[k, 1], y], transform=transform,
                          color=t["muted"], linewidth=0.8, zorder=zorder - 0.5,
                          clip_on=False, gid="edu:leader")
            ax.add_artist(line)
            line.set_visible(bool(np.hypot(x - points[k, 0], y - points[k, 1]) > 0))
            lines.append(line)
    if spread:
        groups = ax.__dict__.setdefault("_edu_badge_groups", [])
        groups.append({"anchors": points, "transform": transform, "badges": badges,
                       "labels": extras, "leaders": lines, "size": size, "dx": dx,
                       "gap": gap})
    return texts + lines


def _estimated_label_widths(labels: Sequence[str] | None, size: float) -> list | None:
    if labels is None:
        return None
    return [(len(str(lab)) * 0.58 * size * 0.9 + 2) if lab else 0.0 for lab in labels]


def _repel_badges(ax: Any, anchors: np.ndarray, transform: Any, sizes: Sequence[float],
                  label_widths_pt: Sequence[float] | None, dx: float, gap: float,
                  radii_px: Sequence[float] | None = None) -> np.ndarray:
    """Positions (in ``transform`` coordinates) for badges at ``anchors``, pushed apart in
    display space inside the axes box; badges already clear of the rest stay put."""
    scale = ax.figure.dpi / 72.0
    disp = transform.transform(anchors)
    radii = np.asarray(radii_px, float) if radii_px is not None else \
        np.array([_badge_radius_pt(s) for s in sizes]) * scale
    widths = None
    if label_widths_pt is not None:
        widths = np.array([(w + abs(dx)) * scale if w else 0.0 for w in label_widths_pt])
    box = ax.bbox
    laid = layout_badges(disp, radii, (box.x0, box.y0, box.x1, box.y1),
                         label_widths=widths, label_sides=[1 if dx >= 0 else -1] * len(disp),
                         gap=gap * scale, square=True)
    return transform.inverted().transform(laid)


def _relayout_badge_groups(fig: Any, renderer: Any) -> None:
    """Spread every ``repel=True`` badge group again at the final panel size."""
    for ax in fig.axes:
        for group in getattr(ax, "_edu_badge_groups", []):
            badges = group["badges"]
            if any(b.axes is not ax for b in badges):
                continue
            anchors = group["anchors"]
            transform = group["transform"]
            try:
                for badge, a in zip(badges, anchors):  # measure the circles at the anchors
                    badge.set_position(tuple(a))
                radii = [_badge_box(b, renderer).width / 2 for b in badges]
                widths = [(lab.get_window_extent(renderer).width * 72.0 / fig.dpi
                           if lab is not None and lab.get_visible() else 0.0)
                          for lab in group["labels"]]
                placed = _repel_badges(ax, anchors, transform, [group["size"]] * len(badges),
                                       widths if any(widths) else None, group["dx"],
                                       group["gap"], radii)
            except Exception:  # noqa: BLE001 - keep the first placement
                continue
            for k, (badge, (x, y)) in enumerate(zip(badges, placed)):
                badge.set_position((x, y))
                lab = group["labels"][k]
                if lab is not None:
                    lab.xy = (x, y)
                if k < len(group["leaders"]):
                    line = group["leaders"][k]
                    a = anchors[k]
                    line.set_data([a[0], x], [a[1], y])
                    moved = np.hypot(*(transform.transform([x, y]) - transform.transform(a)))
                    line.set_visible(bool(moved > 0.5))


def _badge_ticks(ax: Any, positions: Sequence[float], cameras: Sequence[int], *,
                 highlight: Sequence[int] = (), flagged: Sequence[int] = (),
                 size: float = 10, xlabel: str | None = "Camera", axis: str = "x",
                 label: str | None = None) -> None:
    """Replace x (``axis="x"``, poster Fig. 2D) or y (``axis="y"``) tick labels by camera
    badges. ``xlabel`` labels the x axis; ``label`` the y axis (``axis="y"``, default
    none). The layout engine shrinks the badges when they would touch."""
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    texts = [camera_badge_label(c) for c in cameras]
    if axis == "x":
        ax.set_xticks(list(positions))
        ax.set_xticklabels(texts, fontsize=size, fontweight="bold")
        tick_labels = ax.get_xticklabels()
    else:
        ax.set_yticks(list(positions))
        ax.set_yticklabels(texts, fontsize=size, fontweight="bold")
        tick_labels = ax.get_yticklabels()
    ax.tick_params(axis=axis, length=0, pad=size * 0.95)
    for tick_label, cam in zip(tick_labels, cameras):
        tick_label.set_color(badge_text_colour(cam, ax))
        tick_label.set_bbox(_badge_kwargs(ax, cam, size, cam in highlight, cam in flagged))
    if axis == "x" and xlabel:
        ax.set_xlabel(xlabel, labelpad=6)
    if axis == "y" and label:
        ax.set_ylabel(label, labelpad=6)


def badge_ticks_y(ax: Any, positions: Sequence[float], cameras: Sequence[int], *,
                  highlight: Sequence[int] = (), flagged: Sequence[int] = (),
                  size: float = 10, label: str | None = None) -> None:
    """Camera badges as the y tick labels (rows = cameras: matrices, rasters, timelines)."""
    _badge_ticks(ax, positions, cameras, highlight=highlight, flagged=flagged, size=size,
                 axis="y", label=label)


def _camera_items(values_by_camera: Any) -> list[tuple[int, Any]]:
    if isinstance(values_by_camera, Mapping):
        return [(int(k), v) for k, v in values_by_camera.items()]
    return list(enumerate(values_by_camera))


def _nice_ceiling(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    for step in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if step * 10 ** exponent >= value:
            return float(step * 10 ** exponent)
    return float(10 ** (exponent + 1))


def _format_value(value: float) -> str:
    return f"{value:.2g}" if abs(value) < 10 else f"{value:.0f}"


# --------------------------------------------------------------------------------------
# Target faces (UX2-2 / ED2-8, round 2): one face look in every phase

#: Face shades per theme: three *neutral* greys (no hue, so a face can never be read as a
#: camera, whose colours all carry a tint), dark / mid / light, each readable on the stage.
#: Face ``k`` (0-based) uses shade ``k % 3``; the face's identity is its square "F1".."F6"
#: tag (:func:`draw_face_tags`, :func:`face_ticks`) and, for dots, its marker shape
#: (:data:`FACE_MARKERS`). Same face -> same shade, shape and tag in every phase.
#: Rule of thumb for the learner: colour + circle = camera, grey + square tag = face.
FACE_SHADES: dict[str, tuple[str, str, str]] = {
    "Light": ("#2E2E2E", "#6E6E6E", "#A0A0A0"),
    "Sepia": ("#2E2B28", "#6C6863", "#9E9A94"),
    "Dark": ("#F2F2F2", "#B9B9B9", "#808080"),
}
#: Marker per face (scatter ``marker``): never a circle (circles are camera badges / detections).
FACE_MARKERS: tuple[str, ...] = ("s", "^", "D", "v", "p", "h")
FACE_TAG_PREFIX = "F"


def face_label(face: int, *, short: bool = True) -> str:
    """``"F1"`` (``short``) or ``"Face 1"`` for the 0-based ``face`` index (numbered from 1,
    like cameras and photos)."""
    n = int(face) + 1
    return f"{FACE_TAG_PREFIX}{n}" if short else f"Face {n}"


def face_colour(face: int, theme: Any = None) -> str:
    """The shade of 0-based ``face`` on ``theme`` (a theme name, an axes, a figure or None =
    active theme); see :data:`FACE_SHADES`."""
    shades = FACE_SHADES.get(_theme_of(theme), FACE_SHADES["Light"])
    return shades[int(face) % len(shades)]


def face_colours(theme: Any = None, count: int = 6) -> list[str]:
    """``count`` face shades (index = 0-based face), the one face palette of every phase
    (replaces Phase 1's ``face_colours`` and Phase 2's ``_face_colours``)."""
    return [face_colour(k, theme) for k in range(max(int(count), 0))]


def face_marker(face: int) -> str:
    """Scatter marker of 0-based ``face`` (:data:`FACE_MARKERS`)."""
    return FACE_MARKERS[int(face) % len(FACE_MARKERS)]


def face_text_colour(face: int, theme: Any = None) -> str:
    """Black or white tag text on ``face``'s shade."""
    shade = face_colour(face, theme)
    return "#000000" if contrast("#000000", shade) >= contrast(WHITE, shade) else WHITE


def face_point_kwargs(face: int, theme: Any = None, *, size: float = 34.0,
                      on_photo: bool = False) -> dict[str, Any]:
    """``ax.scatter`` kwargs for the corners of one face: its marker in its shade. On a photo
    (``on_photo``) the marker gets a contrasting 1 pt edge (black under light shades, white
    under dark ones) so it reads on bright and dark pixels alike."""
    shade = face_colour(face, theme)
    edge = ("#000000" if relative_luminance(shade) > 0.3 else WHITE) if on_photo else "none"
    return {"s": size, "marker": face_marker(face), "color": shade, "edgecolors": edge,
            "linewidths": 1.0 if on_photo else 0.0}


def face_tag_bbox(face: int, theme: Any = None, *, size: float = 9.0,
                  selected: bool = False) -> dict[str, Any]:
    """bbox dict for a square face tag (``ax.text(..., bbox=face_tag_bbox(...))``); a
    ``selected`` tag gets an accent ring."""
    t = stage_tokens(_theme_of(theme))
    return dict(boxstyle=f"square,pad={0.30 * size / 9:.3f}", facecolor=face_colour(face, theme),
                edgecolor=t["accent"] if selected else "none", linewidth=2.0 if selected else 0.0)


def draw_face_tags(ax: Any, xy: Any, faces: Sequence[int], *, size: float = 9.0,
                   transform: Any = None, selected: Sequence[int] = (), zorder: float = 6,
                   long: bool = False) -> list:
    """Square face tags ("F2") at ``xy`` (N x 2, data coordinates unless ``transform``; on a
    3D axes pass N x 3). ``long`` writes "Face 2". Returns the text artists (gid
    ``"edu:facetag"``)."""
    points = np.asarray(xy, dtype=float)
    points = points.reshape(len(faces), -1)
    texts = []
    for p, face in zip(points, faces):
        kwargs = dict(ha="center", va="center", fontsize=size, fontweight="bold",
                      color=face_text_colour(face, ax), zorder=zorder, gid="edu:facetag",
                      bbox=face_tag_bbox(face, ax, size=size, selected=face in selected))
        label = face_label(face, short=not long)
        if getattr(ax, "name", "") == "3d" and len(p) == 3 and transform is None:
            text = ax.text(p[0], p[1], p[2], label, **kwargs)
        else:
            text = ax.text(p[0], p[1], label, transform=transform or ax.transData,
                           clip_on=False, **kwargs)
        texts.append(text)
    return texts


def face_ticks(ax: Any, positions: Sequence[float], faces: Sequence[int], *, axis: str = "x",
               size: float = 9.0, selected: Sequence[int] = (), label: str | None = None) -> None:
    """Square "F1".."F6" face tags as the tick labels of ``axis`` (matrices: columns = faces)."""
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    texts = [face_label(f) for f in faces]
    if axis == "x":
        ax.set_xticks(list(positions))
        ax.set_xticklabels(texts, fontsize=size, fontweight="bold")
        ticks = ax.get_xticklabels()
    else:
        ax.set_yticks(list(positions))
        ax.set_yticklabels(texts, fontsize=size, fontweight="bold")
        ticks = ax.get_yticklabels()
    ax.tick_params(axis=axis, length=0, pad=size * 0.9)
    for tick, face in zip(ticks, faces):
        tick.set_color(face_text_colour(face, ax))
        tick.set_bbox(face_tag_bbox(face, ax, size=size, selected=face in selected))
    if label:
        (ax.set_xlabel if axis == "x" else ax.set_ylabel)(label, labelpad=6)


def face_legend_handles(faces: Sequence[int], theme: Any = None, *,
                        counts: Mapping[int, int] | None = None) -> list:
    """Legend handles "F2 · 24 corners" with each face's marker and shade."""
    from matplotlib.lines import Line2D
    handles = []
    for face in faces:
        text = face_label(face, short=False)
        if counts is not None and face in counts:
            text += f" · {format_count(counts[face], 'corner')}"
        handles.append(Line2D([], [], lw=0, marker=face_marker(face), markersize=8,
                              markerfacecolor=face_colour(face, theme),
                              markeredgecolor=face_colour(face, theme), label=text))
    return handles


# --------------------------------------------------------------------------------------
# Per-camera plots


def outlier_decimals(values: Sequence[float]) -> int:
    """One decimal count for every ▲ outlier label of a plot (UX-32): 1 when any value is
    below 10, else 0 ("▲ 2.1, 13.8" or "▲ 14, 150", never "▲ 2.1, 14")."""
    vals = [abs(float(v)) for v in values if np.isfinite(v)]
    return 1 if vals and min(vals) < 10 else 0


def _outlier_text(values: Sequence[float], decimals: int, keep: int = 3) -> str:
    shown = list(values)[-keep:]
    text = ", ".join(f"{v:,.{decimals}f}" for v in shown)
    if len(values) > keep:
        text = f"+{len(values) - keep}  " + text
    return f"▲ {text}"


def _outlier_short(values: Sequence[float], decimals: int) -> str:
    """Compact form when the full lists of neighbouring columns would collide."""
    top = f"{max(values):,.{decimals}f}"
    return f"▲ {top}" if len(values) == 1 else f"▲{len(values)} ≤{top}"


#: More round log ticks than this in view are labelled at powers of ten only (the rest keep
#: their gridlines): five crowded labels (3, 10, 30, 100, 300) did not fit a 1280 px panel.
LOG_LABEL_MAX = 4


def _thin_log_labels(ax: Any, scale: Any) -> None:
    """Y-tick thinning for the strip plot's log axis (Phase 3 stage 2 panel C, round 10).
    Only the tick labels change: the tick marks and gridlines stay at every round tick. A
    tick label is written only when it is a power of ten (1, 10, 100 px, ...) and more than
    :data:`LOG_LABEL_MAX` round ticks are in view; other callers with few ticks are untouched."""
    from matplotlib.ticker import FuncFormatter
    lo, hi = ax.get_ylim()
    shown = [tk for tk in scale.ticks if lo * 0.999 <= tk <= hi * 1.001]
    if len(shown) <= LOG_LABEL_MAX:
        return

    def is_decade(value: float) -> bool:
        return value > 0 and abs(float(np.log10(value)) - round(float(np.log10(value)))) < 1e-9

    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _p: f"{v:g}" if is_decade(v) else ""))


def strip_plot_by_camera(ax: Any, values_by_camera: Any, highlight: Any = None,
                         unit: str = "px", *, cap: float | None = None,
                         ylabel: str = "RMS error", xlabel: str | None = "Camera",
                         flagged: Sequence[int] = (), jitter: float = 0.28,
                         seed: int = 7, log: Any = False) -> dict[str, Any]:
    """Poster Fig. 2D: jittered dots per camera in camera colours, badge tick labels.

    ``values_by_camera``: mapping ``camera index -> values`` or a sequence indexed by camera.
    ``highlight``: a camera index (ring on its badge) or ``(camera, value_index)`` (also a red
    ring around that dot, e.g. the image shown in panel A). Values above ``cap`` (automatic
    when ``None``: a nice ceiling over the bulk) are drawn as a triangle above the column with
    their values, in the camera's colour, all with one number format (:func:`outlier_decimals`);
    labels of neighbouring columns that would collide are shortened, then staggered.

    ``log``: ``True`` or an :class:`ErrorScale` puts the values on the shared log error axis
    (round ticks 0.1 0.3 1 3 ... px, limits from :meth:`ErrorScale.log_range`); the
    automatic cap is then the top of that range (values above 100 px become ▲ outliers).
    Returns ``{"cap", "positions", "outliers", "ylim"}``.
    """
    items = _camera_items(values_by_camera)
    t = _tokens_for(ax)
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(v, dtype=float).ravel() for _, v in items]
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays]) if arrays else np.array([])
    scale = None
    if log is not False and log is not None:
        scale = log if isinstance(log, ErrorScale) else error_scale(ax)
    lo = 0.0
    if scale is not None:
        positive = finite[finite > 0]
        lo, top = scale.log_range(positive) if positive.size else scale.clim
        if cap is None:
            cap = top
        if cap <= lo:
            lo = cap / 3
    elif cap is None:
        if finite.size:
            q1, q3, p97 = np.percentile(finite, [25, 75, 97])
            limit = max(q3 + 4.0 * (q3 - q1), 1.8 * p97)
            inliers = finite[finite <= limit]
            cap = _nice_ceiling(float(inliers.max()) if inliers.size else float(limit))
        else:
            cap = 1.0
    hl_cam, hl_idx = (highlight if isinstance(highlight, tuple) else (highlight, None))
    outliers: dict[int, list[float]] = {}
    positions = list(range(len(items)))
    for pos, ((cam, _), values) in enumerate(zip(items, arrays)):
        offsets = rng.uniform(-jitter, jitter, size=values.size)
        inside = np.isfinite(values) & (values <= cap)
        if scale is not None:
            inside &= values > 0
        ax.scatter(pos + offsets[inside], np.maximum(values[inside], lo) if scale is not None
                   else values[inside], s=26, zorder=3,
                   alpha=0.85, **camera_mark_kwargs(cam, ax, "dot"))
        over = values[np.isfinite(values) & (values > cap)]
        if over.size:
            outliers[cam] = sorted(float(v) for v in over)
        if hl_cam is not None and int(hl_cam) == cam and hl_idx is not None \
                and 0 <= int(hl_idx) < values.size:
            y = min(values[int(hl_idx)], cap)
            if scale is not None:
                y = max(y, lo)
            ax.scatter([pos + offsets[int(hl_idx)]], [y], s=150, facecolors="none",
                       edgecolors=t["bad"], linewidths=2.0, zorder=4)
    # Headroom above the cap for the ▲ labels: one row, or two when neighbours both have one.
    cols = [pos for pos, (cam, _) in enumerate(items) if cam in outliers]
    two_rows = any(b - a == 1 for a, b in zip(cols, cols[1:]))
    headroom = (1.26 if two_rows else 1.17) if outliers else (1.0 if scale else 1.04)
    if scale is not None:
        scale.log_axis(ax, "y")
        top = cap * (cap / lo) ** (headroom - 1.0)
        ax.set_ylim(lo, top)
        ax.spines["left"].set_bounds(lo, cap)
        _thin_log_labels(ax, scale)
        cap_frac = 1.0 / headroom
    else:
        ax.set_ylim(0, cap * headroom)
        ax.spines["left"].set_bounds(0, cap)
        ticks = [tick for tick in ax.get_yticks() if 0 <= tick <= cap + 1e-9]
        ax.set_yticks(ticks)
        cap_frac = 1.0 / headroom
    ax.set_xlim(-0.6, len(items) - 0.4)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    if outliers:
        from matplotlib.transforms import blended_transform_factory
        decimals = outlier_decimals([v for vals in outliers.values() for v in vals])
        where = blended_transform_factory(ax.transData, ax.transAxes)
        room = 1.0 - cap_frac
        rows = (cap_frac + room * 0.30, cap_frac + room * 0.72) if two_rows else \
            (cap_frac + room * 0.5,)
        row = 0
        for pos, (cam, _) in enumerate(items):
            if cam not in outliers:
                continue
            if two_rows and pos - 1 in cols:
                row = 1 - row
            else:
                row = 0
            label = ax.text(pos, rows[row], _outlier_text(outliers[cam], decimals),
                            transform=where, ha="center", va="center",
                            fontsize=TICK_SIZE + 1, color=camera_label_colour(ax, cam),
                            fontweight="bold", clip_on=False, gid="edu:outlier")
            label._edu_short = _outlier_short(outliers[cam], decimals)
            label._edu_full = label.get_text()
    ax.set_ylabel(f"{ylabel} ({unit})" if unit else ylabel)
    hl = [int(hl_cam)] if hl_cam is not None else []
    _badge_ticks(ax, positions, [cam for cam, _ in items], highlight=hl, flagged=flagged,
                 xlabel=xlabel)
    return {"cap": cap, "positions": positions, "outliers": outliers,
            "ylim": tuple(ax.get_ylim())}


def bars_by_camera(ax: Any, values: Any, threshold: float | None = None, *,
                   unit: str = "", ylabel: str = "", threshold_label: str | None = None,
                   fail_below: bool = True, flagged: Sequence[int] | None = None,
                   highlight: Any = None, value_labels: bool = True,
                   value_format: str = "{:.3g}", xlabel: str | None = "Camera") -> list[int]:
    """One bar per camera in its colour, badge tick labels, optional dashed threshold line.

    ``values``: mapping ``camera -> value`` or a sequence indexed by camera. Cameras failing the
    threshold (``value < threshold`` when ``fail_below`` else ``value > threshold``), or listed
    in ``flagged``, get a red badge ring and a red bar edge. Returns the flagged camera indices.
    """
    items = [(cam, float(v)) for cam, v in _camera_items(values)]
    t = _tokens_for(ax)
    if flagged is None:
        flagged = []
        if threshold is not None:
            flagged = [cam for cam, v in items
                       if (v < threshold if fail_below else v > threshold)]
    flagged = [int(c) for c in flagged]
    positions = np.arange(len(items))
    heights = [v for _, v in items]
    bars = ax.bar(positions, heights, width=0.62,
                  color=[camera_fill(c, ax) for c, _ in items], zorder=3,
                  edgecolor=[t["bad"] if c in flagged else "none" for c, _ in items],
                  linewidth=[2.0 if c in flagged else 0.0 for c, _ in items])
    top = max([v for v in heights if np.isfinite(v)] + [threshold or 0, 1e-12])
    ax.set_ylim(0, top * 1.18)
    ax.set_xlim(-0.6, len(items) - 0.4)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    if value_labels:
        for bar, (cam, v) in zip(bars, items):
            # gid "edu:value": the layout engine hides a panel's value labels when they
            # would collide (narrow panels, UX-1); the badges always stay.
            ax.annotate(value_format.format(v), (bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=TICK_SIZE, color=t["bad"] if cam in flagged else t["text"],
                        fontweight="bold" if cam in flagged else "normal", gid="edu:value")
    if threshold is not None:
        # UX3-11: red only when a bar fails the threshold, neutral dashes otherwise.
        colour = t["bad"] if flagged else t["muted"]
        ax.axhline(threshold, color=colour, linestyle=(0, (5, 3)), linewidth=1.4, zorder=4)
        if threshold_label:
            ax.annotate(threshold_label, (1.0, threshold), xycoords=("axes fraction", "data"),
                        xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=TICK_SIZE, color=colour, annotation_clip=False)
    if ylabel or unit:
        ax.set_ylabel(f"{ylabel} ({unit})" if unit and ylabel else (ylabel or unit))
    hl = [int(highlight)] if highlight is not None else []
    _badge_ticks(ax, positions, [c for c, _ in items], highlight=hl, flagged=flagged,
                 xlabel=xlabel)
    return flagged


def paired_bars_by_camera(ax: Any, before: Any, after: Any, *,
                          labels: tuple[str, str] = ("before", "after"), unit: str = "px",
                          ylabel: str = "RMS error", xlabel: str | None = "Camera") -> None:
    """Before/after bars per camera (e.g. Phase 3 vs Phase 4 RMS): hatched pale bar, solid bar."""
    b = dict(_camera_items(before))
    a = dict(_camera_items(after))
    cams = sorted(set(b) | set(a))
    t = _tokens_for(ax)
    pos = np.arange(len(cams))
    w = 0.36
    for k, cam in enumerate(cams):
        colour = camera_fill(cam, ax)
        if cam in b:
            ax.bar(pos[k] - w / 2, b[cam], w, color=colour, alpha=0.38, hatch="///",
                   edgecolor=camera_label_colour(ax, cam), linewidth=0.8, zorder=3)
        if cam in a:
            ax.bar(pos[k] + w / 2, a[cam], w, zorder=3, **camera_mark_kwargs(cam, ax))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=t["muted"], alpha=0.38, hatch="///",
                             edgecolor=t["muted"], label=labels[0]),
                       Patch(facecolor=t["muted"], label=labels[1])],
              loc="lower right", ncols=2, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    vals = [v for v in list(b.values()) + list(a.values()) if np.isfinite(v)]
    ax.set_ylim(0, max(vals + [1e-12]) * 1.25)
    ax.set_xlim(-0.6, len(cams) - 0.4)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_ylabel(f"{ylabel} ({unit})" if unit else ylabel)
    _badge_ticks(ax, pos, cams, xlabel=xlabel)


#: Line widths (points) for "thicker = more" encodings (UX-36): thinnest and thickest.
WEIGHT_LINEWIDTHS = (0.8, 6.0)


def weight_linewidths(values: Any, *, lo: float = WEIGHT_LINEWIDTHS[0],
                      hi: float = WEIGHT_LINEWIDTHS[1], vmax: float | None = None) -> np.ndarray:
    """Line widths for counts such as "photos two cameras share" (UX-36: a caption that says
    "thicker = more photos" needs widths a reader can tell apart): square-root scaled from
    ``lo`` (zero) to ``hi`` (``vmax``, default the largest value), so 4x the photos reads
    as 2x the width and a small count is still visible."""
    vals = np.clip(np.asarray(values, dtype=float), 0, None)
    top = float(vmax if vmax is not None else (vals.max() if vals.size else 1.0)) or 1.0
    return lo + (hi - lo) * np.sqrt(np.clip(vals / top, 0, 1))


def weight_legend_handles(values: Sequence[float], colour: str, *, unit: str = "photos",
                          vmax: float | None = None) -> list:
    """Legend handles for :func:`weight_linewidths`: one line per value ("10 photos")."""
    from matplotlib.lines import Line2D
    widths = weight_linewidths(values, vmax=vmax)
    return [Line2D([], [], color=colour, linewidth=float(w),
                   label=f"{format_count(int(round(v)))} {unit}") for v, w in zip(values, widths)]


# --------------------------------------------------------------------------------------
# Images


def _show_image(ax: Any, image: Any, image_size: tuple[int, int] | None) -> tuple[int, int]:
    if image is not None:
        img = np.asarray(image)
        cmap = "gray" if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap, interpolation="antialiased", zorder=0,
                  vmin=None if img.dtype != np.uint8 else 0,
                  vmax=None if img.dtype != np.uint8 else 255)
        h, w = img.shape[:2]
    else:
        if image_size is None:
            raise ValueError("pass image or image_size=(width, height)")
        w, h = image_size
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return w, h


def frame_axes(ax: Any, camera: int | None = None, colour: str | None = None,
               width: float = 2.5) -> None:
    """Draw all four spines as a frame (camera colour when ``camera`` is given), like the poster."""
    colour = colour or (camera_fill(camera, ax) if camera is not None
                        else _tokens_for(ax)["spine"])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(colour)
        spine.set_linewidth(width)


def image_with_points(ax: Any, image: Any, points: Any, colour: str | None = None,
                      rejected: Any = None, *, radius: float | None = None,
                      camera: int | None = None, image_size: tuple[int, int] | None = None,
                      linewidth: float = 1.3) -> Any:
    """Photo with detections as open circles (poster Fig. 2A); ``rejected`` as grey circles.

    ``points``/``rejected`` are N x 2 pixel coordinates (x right, y down). ``radius`` is in
    image pixels (default ~1.1 % of the larger side). ``camera`` frames the photo in that
    camera's colour. ``image=None`` draws on an empty canvas of ``image_size``.
    Returns the detections' collection.
    """
    from matplotlib.collections import EllipseCollection
    w, h = _show_image(ax, image, image_size)
    t = _tokens_for(ax)
    colour = colour or t["accent"]
    r = radius if radius is not None else 0.011 * max(w, h)

    def circles(pts, edge, alpha, lw):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        size = np.full(len(pts), 2 * r)
        coll = EllipseCollection(size, size, np.zeros(len(pts)), units="xy", offsets=pts,
                                 offset_transform=ax.transData, facecolors="none",
                                 edgecolors=edge, linewidths=lw, alpha=alpha, zorder=3)
        ax.add_collection(coll)
        return coll

    if rejected is not None and len(np.asarray(rejected).reshape(-1, 2)):
        circles(rejected, t["rejected"], REJECTED_ALPHA + 0.2, linewidth)
    coll = circles(points, colour, 1.0, linewidth)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    if camera is not None:
        frame_axes(ax, camera)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
    return coll


def _nice_length(target: float) -> float:
    if target <= 0 or not np.isfinite(target):
        return 1.0
    exponent = np.floor(np.log10(target))
    for step in (1, 2, 5):
        if step * 10 ** exponent >= target * 0.7:
            return float(step * 10 ** exponent)
    return float(10 ** (exponent + 1))


def residual_arrows(ax: Any, image: Any, observed: Any, projected: Any, scale: float, *,
                    image_size: tuple[int, int] | None = None, unit: str = "px",
                    cmap: str | None = None, clim: tuple[float, float] | None = None,
                    camera: int | None = None, colour_scale: "ErrorScale | None" = None) -> Any:
    """Residual arrows from each observed point toward its reprojection, magnified ``scale`` x.

    Arrows are coloured by error size (theme-aware viridis, :func:`error_cmap`); observed
    points are small circles in the theme accent; a scale bar in the lower-right reads e.g.
    ``"1 px error (shown x50)"``. ``colour_scale`` (an :class:`ErrorScale`, recommended)
    colours the arrows on the shared log scale; else ``clim`` (default 0..max) is linear.
    Returns the quiver (pass it to :func:`colourbar_px` or ``colour_scale.colourbar``).
    """
    observed = np.asarray(observed, dtype=float).reshape(-1, 2)
    projected = np.asarray(projected, dtype=float).reshape(-1, 2)
    w, h = _show_image(ax, image, image_size)
    t = _tokens_for(ax)
    delta = projected - observed
    size = np.hypot(delta[:, 0], delta[:, 1])
    ax.scatter(observed[:, 0], observed[:, 1], s=14, facecolors="none", edgecolors=t["accent"],
               linewidths=0.9, zorder=3)
    colouring: dict[str, Any]
    if colour_scale is not None:
        colouring = {"cmap": colour_scale.cmap, "norm": colour_scale.norm()}
    else:
        lo, hi = clim if clim is not None else (0.0, float(np.nanmax(size)) if size.size else 1.0)
        colouring = {"cmap": cmap or error_cmap(ax), "clim": (lo, hi)}
    quiver = ax.quiver(observed[:, 0], observed[:, 1], delta[:, 0] * scale, delta[:, 1] * scale,
                       size, angles="xy", scale_units="xy", scale=1,
                       width=0.0042, headwidth=3.6, headlength=4.2, headaxislength=3.8,
                       zorder=4, **colouring)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    # Scale bar: a nice error length whose drawn length is ~12 % of the image width.
    length = _nice_length(0.12 * w / max(scale, 1e-12))
    from matplotlib.offsetbox import AnchoredText
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    t = _tokens_for(ax)
    bar_colour = WHITE if image is not None or is_dark_axes(ax) else t["text"]
    bar = AnchoredSizeBar(ax.transData, length * scale,
                          f"{_format_value(length)} {unit} error (shown ×{scale:g})",
                          "lower right", pad=0.5, borderpad=0.6, sep=4, frameon=True,
                          color=bar_colour, size_vertical=max(h * 0.006, 1.0),
                          fontproperties={"size": TICK_SIZE})
    bar.patch.set_facecolor("#000000" if bar_colour == WHITE else t["stage"])
    bar.patch.set_alpha(0.55)
    bar.patch.set_edgecolor("none")
    ax.add_artist(bar)
    del AnchoredText
    if camera is not None:
        frame_axes(ax, camera)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
    return quiver


def colourbar_px(fig: Any, mappable: Any, ax: Any, label: str = "px", *,
                 shrink: float = 0.82, ticks: int = 3) -> Any:
    """Slim colour bar beside ``ax`` with the unit written above it (poster Fig. 2B)."""
    t = _tokens_for(ax)
    bar = fig.colorbar(mappable, ax=ax, shrink=shrink, aspect=22, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=TICK_SIZE, colors=t["muted"], labelcolor=t["text"], length=3)
    from matplotlib.ticker import MaxNLocator
    bar.locator = MaxNLocator(ticks)
    bar.update_ticks()
    bar.ax.set_title(label, fontsize=LABEL_SIZE, color=t["text"], pad=6, loc="left",
                     fontweight="normal")
    return bar


def empty_panel(ax: Any, message: str) -> None:
    """Centred muted message for a panel with no data (never leave a stage blank)."""
    t = _tokens_for(ax)
    ax.set_axis_off()
    ax.set_xticks([])  # no stray 0..1 tick labels, even if the axis is switched on later
    ax.set_yticks([])
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=LABEL_SIZE + 1,
            color=t["muted"], transform=ax.transAxes)


def integer_axis(ax: Any, axis: str = "y") -> None:
    """Integer ticks only (counts: corners, photos, cameras; UX-21)."""
    from matplotlib.ticker import MaxNLocator
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(MaxNLocator(integer=True, nbins="auto"))


# --------------------------------------------------------------------------------------
# Layout engine: aligned panel titles, elided titles, collision-free value labels/badges

from matplotlib.layout_engine import ConstrainedLayoutEngine  # noqa: E402

#: Smallest font a badge tick may shrink to before the layout gives up (then the checker
#: reports the collision).
MIN_BADGE_SIZE = 7.0


def _ensure_canvas(fig: Any) -> Any:
    """The figure's canvas, attaching an Agg canvas when it has none that can render
    (a bare ``Figure()`` has a ``FigureCanvasBase`` whose ``draw`` does nothing)."""
    canvas = fig.canvas
    if not callable(getattr(canvas, "get_renderer", None)):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = FigureCanvasAgg(fig)
    return canvas


def _renderer_of(fig: Any) -> Any:
    canvas = fig.canvas
    get = getattr(canvas, "get_renderer", None)
    if callable(get):
        return get()
    try:
        return fig._get_renderer()
    except Exception:  # noqa: BLE001 - no renderer at all: draw with Agg
        return _ensure_canvas(fig).get_renderer()


def _edu_title(ax: Any) -> Any:
    for text in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
        if text is not None and (text.get_gid() or "") == "edu:title" and text.get_text():
            return text
    return None


def _subtitle_of(ax: Any) -> Any:
    for text in ax.texts:
        if (text.get_gid() or "") == "edu:subtitle":
            return text
    return None


def _elide_to(text: Any, full: str, limit_px: float, renderer: Any, suffix: str = "") -> None:
    """Set ``text`` to ``full`` (+ ``suffix``), shortened with "…" until its right edge is
    left of ``limit_px`` (display pixels)."""
    text.set_text(full + suffix)
    if text.get_window_extent(renderer).x1 <= limit_px or len(full) < 4:
        return
    lo, hi = 1, len(full) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        text.set_text(full[:mid].rstrip() + "…" + suffix)
        if text.get_window_extent(renderer).x1 <= limit_px:
            lo = mid
        else:
            hi = mid - 1
    text.set_text(full[:lo].rstrip() + "…" + suffix)
    text._edu_elided = True


def _overlapping(boxes: list, pad: float = 0.5) -> bool:
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if (min(a.x1, b.x1) - max(a.x0, b.x0) > pad
                    and min(a.y1, b.y1) - max(a.y0, b.y0) > pad):
                return True
    return False


def _badge_box(label: Any, renderer: Any) -> Any:
    from matplotlib.transforms import Bbox
    box = label.get_window_extent(renderer)
    patch = label.get_bbox_patch()
    if patch is None:
        return box
    grow = 0.35 * label.get_fontsize() * renderer.points_to_pixels(1.0)
    side = max(box.width, box.height) / 2 + grow
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    return Bbox([[cx - side, cy - side], [cx + side, cy + side]])


def _resize_tag(patch: Any, size: float) -> None:
    """New pad for a tick tag at font ``size``, keeping its shape (circle = camera badge,
    square = face tag)."""
    if type(patch.get_boxstyle()).__name__.lower() == "square":
        patch.set_boxstyle(f"square,pad={0.30 * size / 9:.3f}")
    else:
        patch.set_boxstyle(f"circle,pad={0.28 * size / 11:.3f}")


def _fit_badge_ticks(ax: Any, renderer: Any, name: str) -> None:
    """Shrink one axis's badge tick labels until no two touch (down to MIN_BADGE_SIZE)."""
    axis = ax.xaxis if name == "x" else ax.yaxis
    get = ax.get_xticklabels if name == "x" else ax.get_yticklabels

    def badges() -> list:
        return [lab for lab in get() if lab.get_bbox_patch() is not None and lab.get_text()]

    labels = badges()
    if len(labels) < 2:
        return
    attr = "_edu_badge_size" if name == "x" else "_edu_badge_size_y"
    base = getattr(ax, attr, None) or labels[0].get_fontsize()
    setattr(ax, attr, base)
    size = base
    ax.tick_params(axis=name, labelsize=size)
    for lab in get():
        patch = lab.get_bbox_patch()
        if patch is not None:
            _resize_tag(patch, size)
    axis.get_tightbbox(renderer)
    while size > MIN_BADGE_SIZE and _overlapping(
            [_badge_box(lab, renderer) for lab in badges()], 0.0):
        size = max(MIN_BADGE_SIZE, size - 1.0)
        ax.tick_params(axis=name, labelsize=size)
        for lab in get():
            patch = lab.get_bbox_patch()
            if patch is not None:
                _resize_tag(patch, size)
        axis.get_tightbbox(renderer)


def _fit_outlier_labels(ax: Any, renderer: Any) -> None:
    """▲ outlier labels of a strip plot: full lists, or the compact form for every column
    when two would collide, made smaller (to :data:`MIN_BADGE_SIZE`) if even that collides
    -- as it can in a wider font than the one the plot was laid out in."""
    marks = [t for t in ax.texts if (t.get_gid() or "") == "edu:outlier"]
    if len(marks) < 2:
        return
    for t in marks:
        t.set_text(getattr(t, "_edu_full", t.get_text()))
        if not hasattr(t, "_edu_size"):
            t._edu_size = t.get_fontsize()
        t.set_fontsize(t._edu_size)
    if _overlapping([t.get_window_extent(renderer) for t in marks], 0.5):
        for t in marks:
            t.set_text(getattr(t, "_edu_short", t.get_text()))
        size = marks[0].get_fontsize()
        while size > MIN_BADGE_SIZE and _overlapping(
                [t.get_window_extent(renderer) for t in marks], 0.5):
            size = max(MIN_BADGE_SIZE, size - 1.0)
            for t in marks:
                t.set_fontsize(size)


def _fit_value_labels_and_badges(fig: Any, renderer: Any) -> None:
    for ax in fig.axes:
        values = [t for t in ax.texts if (t.get_gid() or "") == "edu:value"]
        for t in values:
            t.set_visible(True)
        if values and _overlapping([t.get_window_extent(renderer) for t in values], 0.5):
            for t in values:
                t.set_visible(False)
        _fit_outlier_labels(ax, renderer)
        _fit_badge_ticks(ax, renderer, "x")
        _fit_badge_ticks(ax, renderer, "y")
    _relayout_badge_groups(fig, renderer)


def _align_and_elide_titles(fig: Any, renderer: Any, align: bool = True) -> None:
    titled = [(ax, _edu_title(ax)) for ax in fig.axes]
    titled = [(ax, t) for ax, t in titled if t is not None and ax.get_visible()]
    if not titled:
        return
    for ax, _ in titled:
        try:
            ax.apply_aspect()
        except Exception:  # noqa: BLE001 - 3D or exotic axes: keep their position
            pass
    # Rows: panels whose layout boxes start within 3 % of the figure height of each other.
    rows: list[list] = []
    for ax, title in sorted(titled, key=lambda m: -m[0].get_position(original=True).y1):
        y1 = ax.get_position(original=True).y1
        if rows and abs(rows[-1][0][0].get_position(original=True).y1 - y1) <= 0.03:
            rows[-1].append((ax, title))
        else:
            rows.append([(ax, title)])
    width_px = fig.bbox.width
    for members in rows:
        if align and len(members) >= 2:
            top = max(ax.get_position().y1 for ax, _ in members)
            for ax, title in members:
                pos = ax.get_position()
                y = (top - pos.y0) / max(pos.height, 1e-9)
                ax._autotitlepos = False
                title.set_y(y)
                sub = _subtitle_of(ax)
                if sub is not None:
                    sub.set_y(y)
        # Elide: each title/subtitle may run to the next titled panel in its row.
        lefts = sorted(ax.get_position().x0 for ax, _ in members)
        for ax, title in members:
            x0 = ax.get_position().x0
            right = [x for x in lefts if x > x0 + 1e-6]
            limit = (right[0] * width_px - 10) if right else width_px - 4
            head = getattr(title, "_edu_head", None)
            if head is not None:
                suffix = "\n " if getattr(title, "_edu_has_subtitle", False) else ""
                _elide_to(title, head, limit, renderer, suffix)
            sub = _subtitle_of(ax)
            if sub is not None and getattr(sub, "_edu_full", None):
                _elide_to(sub, sub._edu_full, limit, renderer)


def finish_panel_layout(fig: Any, renderer: Any = None) -> None:
    """After constrained layout: hide colliding bar value labels, shorten colliding ▲
    outlier labels, fit :func:`card_text` inside its card (UX3-1), shrink colliding badge
    ticks (x and y), spread ``repel=True`` badge
    groups at the final panel size, put every panel title of a row on one baseline (UX-13) and elide titles and
    subtitles that would run into the next panel or off the figure ("…", UX-1)."""
    renderer = renderer or _renderer_of(fig)
    _fit_card_texts(fig, renderer)
    _fit_value_labels_and_badges(fig, renderer)
    _align_and_elide_titles(fig, renderer)


class EduLayoutEngine(ConstrainedLayoutEngine):
    """Constrained layout followed by :func:`finish_panel_layout` on every draw."""

    def execute(self, fig: Any) -> Any:
        # Let constrained layout measure every title at its natural place (an aligned
        # title raised on the previous draw would otherwise claim extra room and push its
        # panel down), then align again.
        for ax in fig.axes:
            title = _edu_title(ax)
            if title is None:
                continue
            sub = _subtitle_of(ax)
            if getattr(ax, "_autotitlepos", None) is False:
                ax._autotitlepos = None
                title.set_y(1.0)
                if sub is not None:
                    sub.set_y(1.0)
        try:  # elide against the current boxes first, so a long title never widens a panel
            _align_and_elide_titles(fig, _renderer_of(fig), align=False)
        except Exception:  # noqa: BLE001
            pass
        info = super().execute(fig)
        try:
            finish_panel_layout(fig)
        except Exception:  # noqa: BLE001 - the picture must still draw
            pass
        return info


def align_panel_titles(fig: Any) -> Any:
    """Switch ``fig`` to :class:`EduLayoutEngine` (constrained layout + aligned, elided
    titles). ``EducationStageView.draw_figure`` does this for every stage figure; call it
    yourself for figures made elsewhere. Returns the engine."""
    engine = EduLayoutEngine()
    fig.set_layout_engine(engine)
    return engine


# --------------------------------------------------------------------------------------
# Text collision checker (tests and review screenshots)


def _short_text(text: str, n: int = 60) -> str:
    plain = " ".join(str(text).split())
    return plain if len(plain) <= n else plain[:n - 1] + "…"


def _figure_texts(fig: Any) -> list[tuple[Any, str, Any]]:
    """(text artist, role, owning axes) for every visible text in ``fig``."""
    out: list[tuple[Any, str, Any]] = [(t, "figure text", None) for t in fig.texts]
    for legend in getattr(fig, "legends", []):
        out += [(t, "legend", None) for t in legend.get_texts()]
    for ax in fig.axes:
        for slot in ("title", "_left_title", "_right_title"):
            t = getattr(ax, slot, None)
            if t is not None:
                out.append((t, "title", ax))
        out += [(t, "subtitle" if (t.get_gid() or "") == "edu:subtitle" else "annotation", ax)
                for t in ax.texts]
        if ax.name == "3d" or not ax.axison:
            # Switched-off axes (axis('off'), empty panels) draw no ticks or axis labels.
            legend = ax.get_legend()
            if legend is not None and legend.get_visible():
                out += [(t, "legend", ax) for t in legend.get_texts()]
                out.append((legend.get_title(), "legend", ax))
            continue
        for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            if not axis.get_visible():
                continue
            out.append((axis.label, f"{name} label", ax))
            out.append((axis.get_offset_text(), f"{name} offset", ax))
            try:
                ticks = axis._update_ticks()
            except Exception:  # noqa: BLE001
                ticks = axis.get_major_ticks()
            for tick in ticks:
                for lab in (tick.label1, tick.label2):
                    if lab.get_visible():
                        out.append((lab, f"{name} tick", ax))
        legend = ax.get_legend()
        if legend is not None and legend.get_visible():
            out += [(t, "legend", ax) for t in legend.get_texts()]
            out.append((legend.get_title(), "legend", ax))
    return out


def text_collisions(fig: Any, *, draw: bool = True, tolerance: float = 1.0) -> list[dict]:
    """Every text problem in a matplotlib figure (UX-1): a text running off the figure
    (clipped) or overlapping another text (titles, subtitles, tick labels, badges, axis
    labels, legends, annotations).

    Draws the figure first (``draw=True``) so the layout is final. A title and its own
    subtitle are one unit and never reported against each other. Returns dicts with
    ``problem`` ("outside figure" | "overlap"), ``text``, ``role``, ``other`` and
    ``panel`` (the panel title, for finding it). Usable in tests
    (``assert not text_collisions(fig)``) and by ``edu_screenshots.py``.
    """
    canvas = _ensure_canvas(fig)
    if draw:
        canvas.draw()
    renderer = _renderer_of(fig)
    fbox = fig.bbox
    items = []
    for text, role, ax in _figure_texts(fig):
        if (not text.get_visible() or not str(text.get_text()).strip()
                or (text.get_alpha() is not None and text.get_alpha() == 0)):
            continue
        if ax is not None and not ax.get_visible():
            continue
        try:
            box = _badge_box(text, renderer) if text.get_bbox_patch() is not None else \
                text.get_window_extent(renderer)
        except Exception:  # noqa: BLE001
            continue
        if box.width <= 0 or box.height <= 0:
            continue
        if role == "title" and "\n" in text.get_text():
            # The blank second line only reserves the subtitle's room.
            lines = text.get_text().count("\n") + 1
            from matplotlib.transforms import Bbox
            box = Bbox([[box.x0, box.y1 - box.height / lines], [box.x1, box.y1]])
        title = _edu_title(ax) if ax is not None else None
        panel = (getattr(title, "_edu_head", None) or (title.get_text() if title else "")) \
            if ax is not None else "figure"
        items.append((text, role, ax, box, panel))
    issues: list[dict] = []
    for text, role, ax, box, panel in items:
        if (box.x0 < fbox.x0 - tolerance or box.x1 > fbox.x1 + tolerance
                or box.y0 < fbox.y0 - tolerance or box.y1 > fbox.y1 + tolerance):
            issues.append({"problem": "outside figure", "text": _short_text(text.get_text()),
                           "role": role, "panel": _short_text(panel, 40), "other": ""})
    for i, (ta, ra, axa, a, pa) in enumerate(items):
        for tb, rb, axb, b, pb in items[i + 1:]:
            if axa is axb and {ra, rb} == {"title", "subtitle"}:
                continue
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > tolerance and h > tolerance:
                issues.append({"problem": "overlap", "text": _short_text(ta.get_text()),
                               "role": ra, "other": _short_text(tb.get_text()),
                               "other_role": rb, "panel": _short_text(pa, 40),
                               "overlap_px": [round(w), round(h)]})
    return issues


#: Smallest drawn size (pixels) of a patch the card check treats as a card.
CARD_MIN_SIZE_PX = (30.0, 14.0)


def _is_card(patch: Any, renderer: Any) -> Any:
    """The display bbox of ``patch`` when it is a visible card (a FancyBboxPatch, or a
    Rectangle with a visible edge) at least :data:`CARD_MIN_SIZE_PX`, else None."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyBboxPatch, Rectangle
    if not patch.get_visible():
        return None
    face = to_rgba(patch.get_facecolor())
    edge = to_rgba(patch.get_edgecolor())
    edged = edge[3] > 0 and patch.get_linewidth() > 0
    if isinstance(patch, FancyBboxPatch):
        if face[3] <= 0 and not edged:
            return None
    elif not (isinstance(patch, Rectangle) and edged):
        return None
    try:
        box = patch.get_window_extent(renderer)
    except Exception:  # noqa: BLE001
        return None
    if box.width < CARD_MIN_SIZE_PX[0] or box.height < CARD_MIN_SIZE_PX[1]:
        return None
    return box


def card_overflows(fig: Any, *, draw: bool = True, tolerance: float = 1.5) -> list[dict]:
    """Text that runs out of the card it is written on (UX3-1, round 3: P4 gate-card
    findings crossing the card border): every visible text whose anchor lies inside a card
    patch (:func:`_is_card`; the smallest one when cards nest) must lie inside that card.
    Returns dicts like :func:`text_collisions` with ``problem`` "outside its card" and
    ``overflow_px``. ``edu_screenshots.py`` adds them to ``figure_text_issues``."""
    canvas = _ensure_canvas(fig)
    if draw:
        canvas.draw()
    renderer = _renderer_of(fig)
    groups = [(None, list(getattr(fig, "patches", [])), list(fig.texts))]
    groups += [(ax, list(ax.patches), list(ax.texts)) for ax in fig.axes if ax.get_visible()]
    issues: list[dict] = []
    for ax, patches, texts in groups:
        cards = [(p, b) for p in patches for b in [_is_card(p, renderer)] if b is not None]
        if not cards:
            continue
        for text in texts:
            if not text.get_visible() or not str(text.get_text()).strip():
                continue
            try:
                anchor = text.get_transform().transform(text.get_position())
                box = _badge_box(text, renderer) if text.get_bbox_patch() is not None else \
                    text.get_window_extent(renderer)
            except Exception:  # noqa: BLE001
                continue
            holding = [(b.width * b.height, b) for _p, b in cards
                       if b.x0 <= anchor[0] <= b.x1 and b.y0 <= anchor[1] <= b.y1]
            if not holding:
                continue
            card = min(holding, key=lambda m: m[0])[1]
            over = max(card.x0 - box.x0, box.x1 - card.x1, card.y0 - box.y0, box.y1 - card.y1)
            if over > tolerance:
                title = _edu_title(ax) if ax is not None else None
                panel = (getattr(title, "_edu_head", None) or (title.get_text() if title else "")
                         ) if ax is not None else "figure"
                issues.append({"problem": "outside its card", "text": _short_text(text.get_text()),
                               "role": "card text", "other": "card border",
                               "panel": _short_text(panel, 40), "overflow_px": round(float(over))})
    return issues


#: Words that say a colour shows an error or a displacement (the error map is right) ...
_ERROR_WORDS = r"\bpx\b|µm|\bum\b|\bmm\b|error|residual|reprojection|displac|pixel"
#: ... and words that say it shows an amount (the error map is wrong).
_AMOUNT_WORDS = (r"count|points per|per cell|share|density|\btie|coupl|how many|number of|"
                 r"fraction|frequency|score|how often")


def _is_hexbin(artist: Any) -> bool:
    from matplotlib.collections import PolyCollection
    if not isinstance(artist, PolyCollection):
        return False
    paths = artist.get_paths()
    return len(paths) == 1 and len(paths[0].vertices) in (6, 7) and \
        len(artist.get_offsets()) > 1


def colour_map_misuse(fig: Any) -> list[dict]:
    """Colour-mapped artists that use the error map (viridis) for data that is not an error,
    where that can be told (UX3-3): an artist tagged as an amount / signed quantity
    (:func:`amount_scale`, :func:`signed_scale`), a hexbin (its colour is a count), a colour
    bar whose unit is not px / µm / mm, or an un-barred panel whose title talks about counts,
    shares, density or ties. Error scales (:class:`ErrorScale`, :class:`LocalScale` in px)
    are never reported. Returns dicts with ``problem``, ``panel``, ``reason``, ``artist``."""
    import re
    error_words = re.compile(_ERROR_WORDS, re.IGNORECASE)
    amount_words = re.compile(_AMOUNT_WORDS, re.IGNORECASE)
    issues: list[dict] = []
    for ax in fig.axes:
        if getattr(ax, "_colorbar", None) is not None or not ax.get_visible():
            continue
        title = _edu_title(ax)
        sub = _subtitle_of(ax)
        head = getattr(title, "_edu_head", None) or (title.get_text() if title else "") or \
            ax.get_title()
        panel = " ".join([str(head), sub.get_text() if sub is not None else ""]).strip()
        mappables = list(ax.images) + [c for c in ax.collections
                                       if c.get_array() is not None and c.get_visible()]
        for artist in mappables:
            if not is_error_cmap(artist.get_cmap()):
                continue
            quantity = getattr(artist, "_edu_quantity", None)
            if quantity == "error":
                continue
            bar = getattr(artist, "colorbar", None)
            bar_text = " ".join([bar.ax.get_title(loc=loc) for loc in ("left", "center", "right")]
                                + [bar.ax.get_ylabel(), bar.ax.get_xlabel()]
                                ).strip() if bar is not None else ""
            reason = None
            if quantity in ("amount", "signed"):
                reason = f"{quantity} data drawn in the error map"
            elif _is_hexbin(artist):
                reason = "density (hexbin counts) drawn in the error map"
            elif bar_text:
                if not error_words.search(bar_text):
                    reason = f"colour bar '{_short_text(bar_text, 30)}' is not an error unit"
            elif amount_words.search(panel) and not error_words.search(panel):
                reason = "panel shows an amount (count / share / density / tie), not an error"
            if reason:
                issues.append({"problem": "error colour map on non-error data",
                               "panel": _short_text(panel or "figure", 50), "reason": reason,
                               "artist": type(artist).__name__})
    return issues


#: Public names for helpers the phase views share (kept private names for compatibility).
tokens_for = _tokens_for
needs_edge = _needs_edge
badge_ticks = _badge_ticks
nice_ceiling = _nice_ceiling

__all__ = [
    "tokens_for", "badge_ticks", "nice_ceiling", "needs_edge",
    "REJECTED_ALPHA", "CONTINUOUS_CMAP", "CAMERA_PALETTE", "camera_colour",
    "camera_badge_label", "badge_text_colour", "camera_label_colour", "relative_luminance", "contrast", "theme_names",
    "resolve_theme", "active_theme_name", "stage_tokens", "theme_colour", "figure_style",
    "axes_theme", "is_dark_axes", "apply_figure_style", "panel_title", "draw_camera_badges",
    "strip_plot_by_camera", "bars_by_camera", "paired_bars_by_camera", "frame_axes",
    "image_with_points", "residual_arrows", "colourbar_px", "empty_panel",
    # round-1 shared layer
    "colour_distance", "MIN_CAMERA_DISTANCE", "MIN_STAGE_CONTRAST",
    "CAMERA_OUTLINE_CONTRAST", "OUTLINE_WIDTH", "camera_needs_outline", "camera_edge",
    "camera_line_effects", "camera_mark_kwargs", "DARK_CMAP_START", "error_cmap",
    "value_dot_kwargs", "ERROR_NAME", "ERROR_UNIT", "ERROR_SCALE_PX", "ERROR_TICKS_PX",
    "error_label", "ErrorScale", "error_scale", "PHOTO_NUMBER_BASE", "photo_number",
    "photo_label", "photos_label", "camera_title", "format_number", "format_px", "format_um",
    "format_count", "format_percent", "format_change", "GLOSSARY", "glossary",
    "glossary_terms_in", "CAPTION_MAX_SENTENCES", "count_sentences", "caption_problems",
    "LETTER_GAP", "integer_axis", "EduLayoutEngine", "align_panel_titles",
    "finish_panel_layout", "text_collisions",
    # round-2 shared layer
    "panel_title_3d", "layout_badges", "badge_ticks_y", "outlier_decimals",
    # round-2 review fixes
    "CAMERA_THEME_FILLS", "camera_fill", "FACE_SHADES", "FACE_MARKERS", "face_label",
    "face_colour", "face_colours", "face_marker", "face_text_colour", "face_point_kwargs",
    "face_tag_bbox", "draw_face_tags", "face_ticks", "face_legend_handles",
    "LOCAL_SCALE_MAX_RATIO", "LOCAL_SCALE_NOTE", "LocalScale", "local_scale",
    "needs_local_scale", "CAPTION_MAX_LINES", "NARROWEST_DIALOG", "format_sig",
    "WEIGHT_LINEWIDTHS", "weight_linewidths", "weight_legend_handles",
    # round-3 review fixes
    "ERROR_MAP_UNITS", "is_error_unit", "viridis_distance", "amount_hue", "amount_cmap",
    "signed_cmap", "is_error_cmap", "amount_scale", "signed_scale", "LIMIT_DASHES",
    "limit_style", "limit_crossed", "limit_line", "cursor_style", "CARD_TEXT_MIN_SIZE",
    "card_text", "card_overflows", "colour_map_misuse", "SUPPORTED_DIALOG_WIDTHS",
]
