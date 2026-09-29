'''Purpose: Per-visual, presentation-only styles for GUI-managed Matplotlib output.
Status: Active; keeps explicit visual overrides outside scientific run artefacts.
Future: Extend to additional visual backends only with separate data-preservation tests.
'''
from __future__ import annotations

import json
from weakref import WeakKeyDictionary
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

SCHEMA = "pycamset.visual-style"
VERSION = 6
# This text registry is the citation source of truth for suggested presets.
# The 2025 Science guide was inspected via its 2026-07-30 Wayback PDF snapshot;
# direct access to the publisher PDF returned 403 during verification.
SUGGESTED_PRESET_REGISTRY_VERSION = "figure-suggestions-v3"
SUGGESTED_PRESET_CITATIONS = (
    "Suggestions only; not journal compliance. Nature: Preparing figures, "
    "https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/ "
    "(accessed 2026-09-23). Science/AAAS: Guide to Preparing Figures (2025), "
    "https://www.science.org/cms/asset/67f37ac8-4d02-4625-8a05-230568cb8323/author_prep_guide_2025.pdf "
    "(archived PDF inspected 2026-07-30; direct publisher fetch returned 403). "
    "DejaVu Sans fallback: DejaVu Fonts licence, https://dejavu-fonts.github.io/License.html. "
    "Cell and IEEE controls are generic starting points without verified journal-specific prescriptions. "
    "Presentation (large) and Grayscale/minimal are lab design presets shared with the optical-mapping "
    "GUI, not journal styles. Typefaces are limited to open-source families with a recorded licence."
)

#: Suggested appearances, shared with the lab's optical-mapping GUI.  Each is a
#: starting point the user can edit, never a claim of journal compliance.  The
#: palette recolours data line series only (see apply_visual_style).
PRESETS: dict[str, dict[str, Any]] = {
    "Nature": {"label": "Nature suggestion", "font_family": "DejaVu Sans", "font_size": 7.0,
               "font_weight": "normal", "line_width": 1.2, "marker_size": 4.5,
               "text_colour": "#000000", "title_colour": "#000000",
               "figure_background": "#ffffff", "axes_background": "#ffffff",
               "series_palette": ["#000000", "#c00000", "#1f4e9c", "#2e7d32", "#e36c0a", "#6a3d9a"]},
    "Science": {"label": "Science suggestion", "font_family": "DejaVu Sans", "font_size": 7.0,
                "font_weight": "normal", "line_width": 1.2, "marker_size": 4.5,
                "text_colour": "#000000", "title_colour": "#000000",
                "figure_background": "#ffffff", "axes_background": "#ffffff",
                "series_palette": ["#000000", "#1f4e9c", "#c00000", "#2e7d32", "#e36c0a", "#6a3d9a"]},
    "Cell": {"label": "Cell suggestion", "font_family": "DejaVu Sans", "font_size": 7.0,
             "font_weight": "normal", "line_width": 1.2, "marker_size": 4.5,
             "text_colour": "#000000", "title_colour": "#000000",
             "figure_background": "#ffffff", "axes_background": "#ffffff",
             "series_palette": ["#000000", "#c00000", "#1f4e9c", "#6a3d9a", "#2e7d32", "#e36c0a"]},
    "IEEE": {"label": "IEEE suggestion", "font_family": "DejaVu Serif", "font_size": 8.0,
             "font_weight": "normal", "line_width": 1.2, "marker_size": 4.5,
             "text_colour": "#000000", "title_colour": "#000000",
             "figure_background": "#ffffff", "axes_background": "#ffffff",
             "series_palette": ["#000000", "#555555", "#8b0000", "#1f4e9c", "#2e7d32", "#4a4a4a"]},
    "Presentation": {"label": "Presentation (large)", "font_family": "DejaVu Sans", "font_size": 18.0,
                     "font_weight": "normal", "line_width": 2.5, "marker_size": 8.0,
                     "text_colour": "#1a1a1a", "title_colour": "#1a1a1a",
                     "figure_background": "#ffffff", "axes_background": "#ffffff",
                     "series_palette": ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]},
    "Grayscale": {"label": "Grayscale/minimal", "font_family": "DejaVu Sans", "font_size": 10.0,
                  "font_weight": "normal", "line_width": 1.2, "marker_size": 4.5,
                  "text_colour": "#000000", "title_colour": "#000000",
                  "figure_background": "#ffffff", "axes_background": "#ffffff",
                  "series_palette": ["#000000", "#444444", "#888888", "#bbbbbb", "#666666", "#999999"]},
}

#: The document identity of the style that applies to every figure without
#: a style of its own ("Save as default for all figures").
DEFAULT_VISUAL_ID = "default:all-figures"
_VISUAL_OVERRIDES = WeakKeyDictionary()
_DETECTION_OVERLAY_BASELINES = WeakKeyDictionary()
_ELEMENT_BASELINES = WeakKeyDictionary()

#: Marker shapes offered for detections and scatter points, shared with the
#: optical-mapping GUI's vocabulary.  "+", "x" and "." are drawn as strokes, so
#: they have no fill; every other shape can be filled or hollow.
MARKER_SHAPES = (("Circle", "o"), ("Square", "s"), ("Diamond", "D"), ("Triangle up", "^"),
                 ("Triangle down", "v"), ("Plus", "+"), ("Cross", "x"), ("Point", "."))
STROKE_MARKERS = frozenset({"+", "x", "."})
LINE_STYLES = (("Solid", "-"), ("Dashed", "--"), ("Dash-dot", "-."), ("Dotted", ":"))

#: Per-element controls: which keys each kind of element accepts.
ELEMENT_KEYS = {
    "line": {"colour", "line_width", "line_style", "marker", "size", "opacity"},
    "reference": {"colour", "line_width", "line_style", "opacity"},
    "bars": {"colour", "edge_colour", "line_width", "opacity"},
    "points": {"colour", "edge_colour", "line_width", "marker", "size", "filled", "opacity"},
}
_LINE_COLOUR_BASELINES = WeakKeyDictionary()


@dataclass
class VisualStyle:
    """Theme-default fields remain unset; values are explicit per-visual overrides."""

    font_family: str | None = None
    font_size: float | None = None
    font_weight: str | None = None
    text_colour: str | None = None
    title_colour: str | None = None
    tick_colour: str | None = None
    axes_colour: str | None = None
    legend_colour: str | None = None
    figure_background: str | None = None
    axes_background: str | None = None
    line_width: float | None = None
    marker_size: float | None = None
    grid_visible: bool | None = None
    legend_visible: bool | None = None
    overlay_size: float | None = None
    overlay_marker: str | None = None
    overlay_colour: str | None = None
    overlay_edge_colour: str | None = None
    overlay_line_width: float | None = None
    overlay_line_style: str | None = None
    overlay_opacity: float | None = None
    overlay_filled: bool | None = None
    series_colours: dict[str, str] = field(default_factory=dict)
    series_styles: dict[str, dict[str, Any]] = field(default_factory=dict)
    colormap: str | None = None
    suggested_preset: str | None = None
    suggested_preset_registry: str | None = None
    series_palette: list[str] | None = None

    def validate(self) -> None:
        """Reject unknown, malformed or out-of-range style state."""
        if self.font_family is not None and (not self.font_family.strip() or len(self.font_family) > 128):
            raise ValueError("font_family must be a non-empty name up to 128 characters")
        if self.font_weight is not None and self.font_weight not in {"normal", "bold"}:
            raise ValueError("font_weight must be 'normal' or 'bold'")
        for name in ("font_size", "marker_size", "overlay_size"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)
                                      or not 1 <= value <= 48):
                raise ValueError(f"{name} must be between 1 and 48")
        if self.line_width is not None and (
                not isinstance(self.line_width, (int, float))
                or isinstance(self.line_width, bool) or not 0.2 <= self.line_width <= 12):
            raise ValueError("line_width must be between 0.2 and 12")
        if self.overlay_line_width is not None and (
                not isinstance(self.overlay_line_width, (int, float))
                or isinstance(self.overlay_line_width, bool)
                or not 0.2 <= self.overlay_line_width <= 12):
            raise ValueError("overlay_line_width must be between 0.2 and 12")
        for name in ("text_colour", "title_colour", "tick_colour", "axes_colour",
                     "legend_colour", "figure_background", "axes_background",
                     "overlay_colour", "overlay_edge_colour"):
            _validate_colour(getattr(self, name), name)
        if self.overlay_marker is not None and self.overlay_marker not in {"o", "s", "^", "v", "D", "+", "x", "."}:
            raise ValueError("unsupported detection overlay marker")
        if self.overlay_line_style is not None and self.overlay_line_style not in {"-", "--", "-.", ":"}:
            raise ValueError("unsupported detection overlay line style")
        if self.overlay_filled is not None and not isinstance(self.overlay_filled, bool):
            raise ValueError("overlay_filled must be boolean or null")
        if self.overlay_opacity is not None and (
                not isinstance(self.overlay_opacity, (int, float))
                or isinstance(self.overlay_opacity, bool) or not 0 <= self.overlay_opacity <= 1):
            raise ValueError("overlay_opacity must be between 0 and 1")
        if len(self.series_colours) > 128:
            raise ValueError("too many series colour overrides")
        for series_id, colour in self.series_colours.items():
            if not isinstance(series_id, str) or not series_id or len(series_id) > 160:
                raise ValueError("series IDs must be non-empty strings up to 160 characters")
            _validate_colour(colour, f"series_colours[{series_id!r}]")
        if len(self.series_styles) > 128:
            raise ValueError("too many per-series style overrides")

        for series_id, values in self.series_styles.items():
            if not isinstance(series_id, str) or not series_id or len(series_id) > 160 or not isinstance(values, dict):
                raise ValueError("series style entries require stable IDs and objects")
            # Element IDs carry their kind ("reference:0:#1", "bars:0:#1", ...);
            # anything else is a data line, as before.
            kind = series_id.split(":", 1)[0]
            allowed = ELEMENT_KEYS.get(kind, ELEMENT_KEYS["line"])
            if set(values) - allowed:
                raise ValueError("unsupported per-series style control")
            if "line_width" in values and (not isinstance(values["line_width"], (int, float))
                    or isinstance(values["line_width"], bool) or not 0.2 <= values["line_width"] <= 12):
                raise ValueError("series line_width must be between 0.2 and 12")
            if values.get("line_style", "-") not in {"-", "--", "-.", ":"}:
                raise ValueError("unsupported series line_style")
            if values.get("marker", "") not in {"", "o", "s", "^", "v", "D", "+", "x", "."}:
                raise ValueError("unsupported series marker")
            if "size" in values and (not isinstance(values["size"], (int, float))
                    or isinstance(values["size"], bool) or not 1 <= values["size"] <= 48):
                raise ValueError("series size must be between 1 and 48")
            if "opacity" in values and (not isinstance(values["opacity"], (int, float))
                    or isinstance(values["opacity"], bool) or not 0 <= values["opacity"] <= 1):
                raise ValueError("series opacity must be between 0 and 1")
            if "filled" in values and not isinstance(values["filled"], bool):
                raise ValueError("series filled must be true or false")
            for key in ("colour", "edge_colour"):
                if key in values:
                    _validate_colour(values[key], f"series_styles[{series_id!r}].{key}")
        if self.colormap is not None and self.colormap not in {"viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdBu_r"}:
            raise ValueError("unsupported colormap")
        if self.suggested_preset is not None and self.suggested_preset not in PRESETS:
            raise ValueError("unsupported suggested preset")
        if self.series_palette is not None:
            if not isinstance(self.series_palette, list) or not 1 <= len(self.series_palette) <= 12:
                raise ValueError("series_palette must list 1 to 12 colours")
            for index, colour in enumerate(self.series_palette):
                _validate_colour(colour, f"series_palette[{index}]")
        if self.suggested_preset is None and self.suggested_preset_registry is not None:
            raise ValueError("preset registry requires a suggested preset")
        if self.suggested_preset_registry is not None and (
                not isinstance(self.suggested_preset_registry, str)
                or not self.suggested_preset_registry
                or len(self.suggested_preset_registry) > 80):
            raise ValueError("invalid suggested preset registry revision")
        if self.grid_visible is not None and not isinstance(self.grid_visible, bool):
            raise ValueError("grid_visible must be boolean or null")
        if self.legend_visible is not None and not isinstance(self.legend_visible, bool):
            raise ValueError("legend_visible must be boolean or null")


def _validate_colour(value: Any, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or len(value) != 7 or value[0] != "#":
        raise ValueError(f"{name} must be #RRGGBB or null")
    try:
        int(value[1:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be #RRGGBB or null") from exc


def style_to_json(style: VisualStyle, visual_id: str) -> str:
    """Serialise one visual's style, rejecting invalid state before writing."""
    style.validate()
    return json.dumps({"schema": SCHEMA, "version": VERSION, "visual_id": visual_id,
                       "style": asdict(style)}, indent=2, sort_keys=True) + "\n"


def style_from_json(text: str, expected_visual_id: str | None = None) -> VisualStyle:
    """Parse a versioned style document and fail closed on any unknown key."""
    try:
        document = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid style JSON: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {"schema", "version", "visual_id", "style"}:
        raise ValueError("Style document has missing or unknown top-level keys")
    if document["schema"] != SCHEMA or type(document["version"]) is not int or document["version"] not in {1, 2, 3, 4, 5, VERSION}:
        raise ValueError("Unsupported style schema or version")
    if not isinstance(document["visual_id"], str) or not document["visual_id"]:
        raise ValueError("visual_id must be a non-empty string")
    if expected_visual_id is not None and document["visual_id"] != expected_visual_id:
        raise ValueError("Style belongs to a different visual")
    values = document["style"]
    if not isinstance(values, dict):
        raise ValueError("Style has missing or unknown keys")
    if document["version"] == 1:
        # V1 documents are migrated in memory; the original JSON remains untouched
        # until a user explicitly saves the upgraded style.
        values = {**values, "font_weight": None, "title_colour": None, "tick_colour": None,
                  "axes_colour": None, "legend_colour": None, "series_styles": {},
                  "colormap": None, "suggested_preset": None, "overlay_marker": None,
                  "overlay_line_width": None, "overlay_line_style": None, "overlay_opacity": None}
    if document["version"] in {1, 2}:
        # Older files keep their preset name; provenance is unknown rather than guessed.
        values = {**values, "suggested_preset_registry": None,
                  "overlay_marker": None, "overlay_line_width": None,
                  "overlay_line_style": None, "overlay_opacity": None}
    if document["version"] == 3:
        values = {**values, "overlay_marker": None, "overlay_line_width": None,
                  "overlay_line_style": None, "overlay_opacity": None}
    if document["version"] < 5:
        values = {**values, "series_palette": None}
    if document["version"] < 6:
        values = {**values, "overlay_filled": None}
    if set(values) != set(VisualStyle.__dataclass_fields__):
        raise ValueError("Style has missing or unknown keys")
    style = VisualStyle(**values)
    style.validate()
    return style


def _plain_label(artist: Any) -> str | None:
    """A user-given label, or None for Matplotlib's automatic '_child0' names."""
    label = artist.get_label() if hasattr(artist, "get_label") else None
    return label if label and not str(label).startswith("_") else None


def _axes_elements(axes: Any, axes_index: int) -> list[tuple[str, str, str, Any]]:
    """List one axes' styleable elements as (id, kind, display name, artist).

    IDs are stable across sessions for the same figure: a producer's gid if it
    set one, else the element's label, else its position among unlabelled
    elements of its kind.  Data lines keep the "line:<axes>:<label>" IDs that
    earlier style files used.
    """
    from matplotlib.container import BarContainer
    from matplotlib.collections import PathCollection

    elements = []
    counters = {"line": 0, "reference": 0, "bars": 0, "points": 0}

    def element_id(kind, artist, label):
        counters[kind] += 1
        gid = artist.get_gid() if hasattr(artist, "get_gid") else None
        if gid:
            return gid
        return f"{kind}:{axes_index}:{label}" if label else f"{kind}:{axes_index}:#{counters[kind]}"

    for line in axes.lines:
        label = _plain_label(line)
        kind = "line" if line.get_transform() is axes.transData else "reference"
        name = label or ("Line" if kind == "line" else "Reference line")
        elements.append((element_id(kind, line, label), kind, name, line))
    for container in axes.containers:
        if isinstance(container, BarContainer):
            label = _plain_label(container)
            elements.append((element_id("bars", container, label), "bars", label or "Bars", container))
    for collection in axes.collections:
        if isinstance(collection, PathCollection) and not (collection.get_gid() or "").startswith("detection-overlay:"):
            label = _plain_label(collection)
            elements.append((element_id("points", collection, label), "points", label or "Points", collection))
    # Unlabelled elements of one kind get numbered names so rows are distinct.
    seen: dict[str, int] = {}
    named = []
    for element_id_, kind, name, artist in elements:
        seen[name] = seen.get(name, 0) + 1
        named.append((element_id_, kind, name, artist))
    totals = {name: count for name, count in seen.items()}
    running: dict[str, int] = {}
    result = []
    for element_id_, kind, name, artist in named:
        if totals[name] > 1:
            running[name] = running.get(name, 0) + 1
            name = f"{name} {running[name]}"
        result.append((element_id_, kind, name, artist))
    return result


def styleable_elements(figure: Any) -> list[dict[str, Any]]:
    """Every element a user can restyle, for the style dialog's element table."""
    rows = []
    multiple_axes = len(figure.axes) > 1
    for axes_index, axes in enumerate(figure.axes):
        for element_id, kind, name, artist in _axes_elements(axes, axes_index):
            title = axes.get_title() if multiple_axes else ""
            rows.append({"id": element_id, "kind": kind, "artist": artist,
                         "name": f"{name} ({title})" if title else name})
    if any((collection.get_gid() or "").startswith("detection-overlay:")
           for axes in figure.axes for collection in axes.collections):
        rows.append({"id": "detections", "kind": "detections", "artist": None,
                     "name": "Detection markers"})
    return rows


def _element_baseline(figure: Any, artist: Any) -> dict[str, Any]:
    """Capture an element's producer appearance once, for restoring later."""
    store = _ELEMENT_BASELINES.setdefault(figure, WeakKeyDictionary())
    key = artist.patches[0] if hasattr(artist, "patches") and artist.patches else artist
    if key not in store:
        if hasattr(artist, "patches"):
            store[key] = {"patches": [(patch, patch.get_facecolor(), patch.get_edgecolor(),
                                       patch.get_linewidth(), patch.get_alpha())
                                      for patch in artist.patches]}
        elif hasattr(artist, "get_sizes"):
            store[key] = {"face": artist.get_facecolors().copy(), "edge": artist.get_edgecolors().copy(),
                          "sizes": artist.get_sizes().copy(), "paths": artist.get_paths(),
                          "linewidths": artist.get_linewidths().copy(), "alpha": artist.get_alpha()}
        else:
            store[key] = {"colour": artist.get_color(), "width": artist.get_linewidth(),
                          "style": artist.get_linestyle(), "alpha": artist.get_alpha(),
                          "marker_size": artist.get_markersize()}
    return store[key]


def _marker_path(marker: str):
    from matplotlib.markers import MarkerStyle

    shape = MarkerStyle(marker)
    return shape.get_path().transformed(shape.get_transform())


def _style_markers(collection: Any, *, colour: str | None, edge: str | None, filled: bool | None,
                   marker: str | None, size: float | None, width: float | None,
                   alpha: float | None, baseline: dict[str, Any],
                   default_face: Any, default_edge: Any) -> None:
    """Style a marker collection; hollow and stroke shapes are outlined in the main colour."""
    collection.set_paths([_marker_path(marker)] if marker else baseline["paths"])
    collection.set_sizes([size ** 2] if size is not None else baseline["sizes"])
    face = colour or default_face
    stroke = marker in STROKE_MARKERS
    if stroke or filled is False:
        # No fill: the outline carries the colour, and must be wide enough to see.
        collection.set_facecolor("none")
        collection.set_edgecolor(colour or (default_face if isinstance(default_face, str) else baseline["face"]))
        collection.set_linewidths([width] if width is not None
                                  else [max(1.2, float(max(baseline["linewidths"], default=0.0)))])
    else:
        collection.set_facecolor(face)
        collection.set_edgecolor(edge or default_edge)
        collection.set_linewidths([width] if width is not None else baseline["linewidths"])
    collection.set_alpha(alpha if alpha is not None else baseline["alpha"])


def _style_elements(figure: Any, axes: Any, axes_index: int, style: VisualStyle) -> None:
    """Apply per-element settings to reference lines, bars and scatter points.

    An element without settings is returned to its producer's appearance, so
    clearing a setting in the dialog really removes it.
    """
    for element_id, kind, _name, artist in _axes_elements(axes, axes_index):
        if kind == "line":
            continue  # data lines are styled with the series controls
        settings = style.series_styles.get(element_id, {})
        if kind == "reference" and not settings:
            # Older style files addressed labelled reference lines as data lines.
            settings = style.series_styles.get(f"line:{axes_index}:{_plain_label(artist)}", {})
        baseline = _element_baseline(figure, artist)
        if kind == "reference":
            artist.set_color(settings.get("colour", baseline["colour"]))
            if "line_width" in settings:
                artist.set_linewidth(settings["line_width"])
            elif style.line_width is None:
                artist.set_linewidth(baseline["width"])
            artist.set_linestyle(settings.get("line_style", baseline["style"]))
            artist.set_alpha(settings.get("opacity", baseline["alpha"]))
        elif kind == "bars":
            for patch, face, edge, width, alpha in baseline["patches"]:
                patch.set_facecolor(settings.get("colour", face))
                patch.set_edgecolor(settings.get("edge_colour", edge))
                patch.set_linewidth(settings.get("line_width", width))
                patch.set_alpha(settings.get("opacity", alpha))
        elif kind == "points":
            _style_markers(artist, colour=settings.get("colour"), edge=settings.get("edge_colour"),
                           filled=settings.get("filled"), marker=settings.get("marker") or None,
                           size=settings.get("size"), width=settings.get("line_width"),
                           alpha=settings.get("opacity"), baseline=baseline,
                           default_face=baseline["face"], default_edge=baseline["edge"])


def apply_visual_style(figure: Any, style: VisualStyle, theme_name: str = "Light") -> None:
    """Apply presentation properties only, leaving data, limits and colormaps intact."""
    from pyCamSet.gui.theme import THEME_TOKENS

    style.validate()
    if theme_name not in THEME_TOKENS:
        raise ValueError(f"Unknown theme: {theme_name}")
    if style.colormap is not None and not any(
            (image.get_gid() or "").startswith("style:colormap:")
            for axes in figure.axes for image in axes.images):
        raise ValueError("No image explicitly supports a cosmetic colormap override")
    _VISUAL_OVERRIDES[figure] = style
    tokens = THEME_TOKENS[theme_name]
    # Keep each producer's original marker geometry/style so clearing an
    # override restores that figure's real baseline rather than guessed defaults.
    overlay_baselines = _DETECTION_OVERLAY_BASELINES.setdefault(figure, WeakKeyDictionary())
    for axes in figure.axes:
        for collection in axes.collections:
            if (collection.get_gid() or "").startswith("detection-overlay:") \
                    and collection not in overlay_baselines:
                overlay_baselines[collection] = {
                    "sizes": collection.get_sizes().copy() if hasattr(collection, "get_sizes") else None,
                    "paths": collection.get_paths() if hasattr(collection, "get_paths") else None,
                    "linewidths": collection.get_linewidths().copy() if hasattr(collection, "get_linewidths") else None,
                    "linestyles": collection.get_linestyles() if hasattr(collection, "get_linestyles") else None,
                    "alpha": collection.get_alpha() if hasattr(collection, "get_alpha") else None,
                }
    line_baselines = _LINE_COLOUR_BASELINES.setdefault(figure, WeakKeyDictionary())
    figure.set_facecolor(style.figure_background or tokens["surface"])
    for axes_index, axes in enumerate(figure.axes):
        data_line_index = 0
        line_ids = {artist: element_id for element_id, kind, _name, artist
                    in _axes_elements(axes, axes_index) if kind == "line"}
        axes.set_facecolor(style.axes_background or tokens["surface"])
        axes.title.set_color(style.text_colour or tokens["text"])
        if style.title_colour is not None:
            axes.title.set_color(style.title_colour)
        axes.xaxis.label.set_color(style.text_colour or tokens["text"])
        axes.yaxis.label.set_color(style.text_colour or tokens["text"])
        for tick in axes.get_xticklabels() + axes.get_yticklabels():
            tick.set_color(style.tick_colour or style.text_colour or tokens["text"])
        if style.axes_colour is not None:
            axes.xaxis.label.set_color(style.axes_colour)
            axes.yaxis.label.set_color(style.axes_colour)
        if style.grid_visible is not None:
            axes.grid(style.grid_visible)
        for line in axes.lines:
            if line not in line_baselines:
                line_baselines[line] = line.get_color()
            if style.line_width is not None:
                line.set_linewidth(style.line_width)
            if style.marker_size is not None:
                line.set_markersize(style.marker_size)
            series_id = line_ids.get(line)
            # A preset palette recolours data series only: lines drawn in data
            # coordinates.  Reference lines (axhline/axvline thresholds use a
            # blended transform) keep their meaning-carrying colours, and
            # bars, scatters and images are never touched.
            if line.get_transform() is axes.transData:
                if style.series_palette:
                    line.set_color(style.series_palette[data_line_index % len(style.series_palette)])
                else:
                    line.set_color(line_baselines[line])
                data_line_index += 1
            if series_id in style.series_colours:
                line.set_color(style.series_colours[series_id])
            series = style.series_styles.get(series_id, {})
            if "line_width" in series:
                line.set_linewidth(series["line_width"])
            if "line_style" in series:
                line.set_linestyle(series["line_style"])
            if "marker" in series:
                line.set_marker(series["marker"] or "None")
            if "colour" in series:
                line.set_color(series["colour"])
            if "size" in series:
                line.set_markersize(series["size"])
            if series_id is not None and line.get_transform() is axes.transData:
                line.set_alpha(series.get("opacity", _element_baseline(figure, line)["alpha"]))
        _style_elements(figure, axes, axes_index, style)
        # Detection overlays opt in through a stable gid and are styled without
        # touching the underlying image or detection coordinate arrays.
        for collection in axes.collections:
            gid = collection.get_gid() or ""
            if gid.startswith("detection-overlay:"):
                baseline = {**overlay_baselines[collection],
                            "face": collection.get_facecolors().copy()}
                _style_markers(collection, colour=style.overlay_colour, edge=style.overlay_edge_colour,
                               filled=style.overlay_filled, marker=style.overlay_marker,
                               size=style.overlay_size, width=style.overlay_line_width,
                               alpha=style.overlay_opacity, baseline=baseline,
                               default_face=tokens["accent"], default_edge=tokens["border_strong"])
                collection.set_linestyles(style.overlay_line_style or baseline["linestyles"])
        legend = axes.get_legend()
        if legend is not None:
            _sync_legend_swatches(axes, legend)
            if style.legend_visible is not None:
                legend.set_visible(style.legend_visible)
            if style.legend_colour is not None or style.text_colour is not None:
                for text in legend.get_texts():
                    text.set_color(style.legend_colour or style.text_colour)
        # Scalar colour maps are deliberately limited to explicitly opted-in
        # image artists. Existing semantic maps/ranges are never themed.
        if style.colormap is not None:
            opted_in = [image for image in axes.images
                        if (image.get_gid() or "").startswith("style:colormap:")]
            if not opted_in:
                raise ValueError("No image explicitly supports a cosmetic colormap override")
            for image in opted_in:
                image.set_cmap(style.colormap)
    # Font properties are applied to existing text artists, not data artists.
    # The family always resolves to an approved open-source typeface.
    from pyCamSet.gui.figure_fonts import resolve_figure_font
    font_family = resolve_figure_font(style.font_family)
    for text in figure.findobj(match=lambda artist: hasattr(artist, "set_fontsize")):
        if style.font_size is not None:
            text.set_fontsize(style.font_size)
        if font_family is not None:
            text.set_fontfamily(font_family)
        if style.font_weight is not None:
            text.set_fontweight(style.font_weight)
    figure.canvas.draw_idle()


def _sync_legend_swatches(axes: Any, legend: Any) -> None:
    """Make each legend swatch match its restyled line.

    Legend entries are separate proxy artists, so recolouring a line leaves
    its swatch showing the old colour.  Entries are matched to lines by
    label; entries without a matching line are left alone.
    """
    lines = {line.get_label(): line for line in axes.lines}
    handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
    for text, handle in zip(legend.get_texts(), handles):
        line = lines.get(text.get_text())
        if line is None or not hasattr(handle, "set_linestyle"):
            continue
        handle.set_color(line.get_color())
        handle.set_linewidth(line.get_linewidth())
        handle.set_linestyle(line.get_linestyle())


def refresh_visual_style(figure: Any, theme_name: str) -> None:
    """Reapply explicit overrides after the GUI's semantic theme refresh."""
    style = _VISUAL_OVERRIDES.get(figure)
    if style is not None:
        apply_visual_style(figure, style, theme_name)


def _capture_presentation_state(figure: Any) -> dict[str, Any]:
    """Snapshot only properties the style applier can change, for cancel rollback."""
    state: dict[str, Any] = {
        "figure": figure.get_facecolor(),
        "axes": [],
        "text_artists": [(artist, artist.get_color(), artist.get_fontsize(),
                           artist.get_fontfamily(), artist.get_fontweight())
                         for artist in figure.findobj(
                             match=lambda item: hasattr(item, "set_fontsize")
                             and hasattr(item, "get_color"))],
    }
    for axes in figure.axes:
        axes_state = {
            "axes_object": axes,
            "axes": axes.get_facecolor(),
            "lines": [(line, line.get_linewidth(), line.get_markersize(), line.get_color(),
                       line.get_linestyle(), line.get_marker(), line.get_alpha())
                      for line in axes.lines],
            "patches": [(patch, patch.get_facecolor(), patch.get_edgecolor(),
                         patch.get_linewidth(), patch.get_alpha())
                        for container in axes.containers if hasattr(container, "patches")
                        for patch in container.patches],
            "images": [(image, image.get_cmap().copy()) for image in axes.images],
            "collections": [(collection,
                             collection.get_sizes().copy() if hasattr(collection, "get_sizes") else None,
                             collection.get_facecolors().copy() if hasattr(collection, "get_facecolors") else None,
                             collection.get_edgecolors().copy() if hasattr(collection, "get_edgecolors") else None,
                             collection.get_paths() if hasattr(collection, "get_paths") else None,
                             collection.get_linewidths().copy() if hasattr(collection, "get_linewidths") else None,
                             collection.get_linestyles() if hasattr(collection, "get_linestyles") else None,
                             collection.get_alpha() if hasattr(collection, "get_alpha") else None)
                            for collection in axes.collections
                            if hasattr(collection, "get_sizes")],
            "grid": [(gridline, gridline.get_visible())
                     for gridline in axes.xaxis.get_gridlines() + axes.yaxis.get_gridlines()],
            "legend": axes.get_legend(),
        }
        legend = axes_state["legend"]
        axes_state["legend_visible"] = legend.get_visible() if legend is not None else None
        axes_state["legend_text"] = ([(text, text.get_color()) for text in legend.get_texts()]
                                      if legend is not None else [])
        state["axes"].append(axes_state)
    return state


def _restore_presentation_state(figure: Any, state: dict[str, Any]) -> None:
    """Restore the captured rendered appearance without altering scientific data."""
    figure.set_facecolor(state["figure"])
    for axes_state in state["axes"]:
        axes = axes_state["axes_object"]
        axes.set_facecolor(axes_state["axes"])
        for line, width, marker_size, colour, linestyle, marker, alpha in axes_state["lines"]:
            line.set_linewidth(width)
            line.set_markersize(marker_size)
            line.set_color(colour)
            line.set_linestyle(linestyle)
            line.set_marker(marker)
            line.set_alpha(alpha)
        for patch, face, edge, width, alpha in axes_state["patches"]:
            patch.set_facecolor(face)
            patch.set_edgecolor(edge)
            patch.set_linewidth(width)
            patch.set_alpha(alpha)
        for image, cmap in axes_state["images"]:
            image.set_cmap(cmap)
        for collection, sizes, face, edge, paths, linewidths, linestyles, alpha in axes_state["collections"]:
            if sizes is not None:
                collection.set_sizes(sizes)
            if face is not None:
                collection.set_facecolor(face)
            if edge is not None:
                collection.set_edgecolor(edge)
            if paths is not None:
                collection.set_paths(paths)
            if linewidths is not None:
                collection.set_linewidths(linewidths)
            if linestyles is not None:
                collection.set_linestyles(linestyles)
            collection.set_alpha(alpha)
        for gridline, visible in axes_state["grid"]:
            gridline.set_visible(visible)
        legend = axes_state["legend"]
        if legend is not None:
            legend.set_visible(axes_state["legend_visible"])
            for artist, colour in axes_state["legend_text"]:
                artist.set_color(colour)
    for artist, colour, size, family, weight in state["text_artists"]:
        artist.set_color(colour)
        artist.set_fontsize(size)
        artist.set_fontfamily(family)
        artist.set_fontweight(weight)
    figure.canvas.draw_idle()


def scale_bar_unavailable(pixel_to_world: Any = None, unit: str | None = None) -> str | None:
    """Keep scale bars unavailable until a typed, view-specific calibration contract exists."""
    return "Scale bar unavailable: a supported view-specific pixel-to-world calibration contract is not defined."


def style_path_for_visual(app_config_dir: Path, visual_id: str) -> Path:
    """Return an isolated presentation-settings path (never a run artefact path)."""
    import hashlib

    digest = hashlib.sha256(visual_id.encode("utf-8")).hexdigest()[:16]
    return app_config_dir / "visual-styles" / f"{digest}.json"


def default_style_path(app_config_dir: Path) -> Path:
    """Where the style for every figure without its own style is kept."""
    return app_config_dir / "visual-styles" / "default.json"


def as_default_style(style: VisualStyle) -> VisualStyle:
    """Strip what only makes sense for one figure (its series and colour map)."""
    return replace(style, series_colours={}, series_styles={}, colormap=None)


def load_default_style(app_config_dir: Path) -> VisualStyle | None:
    """Return the saved default style, or None when there is none or it is invalid."""
    path = default_style_path(app_config_dir)
    try:
        return style_from_json(path.read_text(encoding="utf-8"), DEFAULT_VISUAL_ID)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        # Fail closed: a damaged default never half-applies; the theme wins.
        return None


def save_default_style(app_config_dir: Path, style: VisualStyle) -> Path:
    """Atomically store *style* as the default for all figures."""
    import tempfile

    path = default_style_path(app_config_dir)
    text = style_to_json(as_default_style(style), DEFAULT_VISUAL_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)
    return path


def clear_default_style(app_config_dir: Path) -> None:
    """Remove the default, so figures without their own style follow the theme."""
    default_style_path(app_config_dir).unlink(missing_ok=True)


def load_style_for_visual(app_config_dir: Path, visual_id: str,
                          legacy_path: Path | None = None) -> tuple[VisualStyle, str]:
    """Resolve a figure's style: its own, else the saved default, else the theme.

    Returns the style and where it came from: ``"visual"``, ``"default"`` or
    ``"theme"``.  A malformed per-figure file falls back rather than failing.
    """
    own = style_path_for_visual(app_config_dir, visual_id)
    candidates = [(own, visual_id)]
    if legacy_path is not None and not own.exists():
        candidates.append((legacy_path, None))
    for path, expected in candidates:
        if path.exists():
            try:
                return style_from_json(path.read_text(encoding="utf-8"), expected), "visual"
            except (OSError, ValueError):
                break
    default = load_default_style(app_config_dir)
    if default is not None:
        return default, "default"
    return VisualStyle(), "theme"


def _validate_user_style_filename(path: str) -> None:
    """Reject filenames that Windows reserves, regardless of the current OS."""
    if "\0" in path:
        raise ValueError("The filename must not contain a NUL character.")
    # QFileDialog returns native paths; split both separators so validation
    # remains consistent when a style path was selected on another platform.
    basename = path.replace("\\", "/").rsplit("/", 1)[-1]
    device_name = basename.split(".", 1)[0].rstrip(" .").casefold()
    reserved = {"con", "prn", "aux", "nul"}
    reserved.update(f"{prefix}{number}" for prefix in ("com", "lpt")
                    for number in "123456789¹²³")
    if device_name in reserved:
        raise ValueError(f"'{basename}' is a reserved filename on Windows.")


class VisualStyleDialog:
    """Small optional-import Qt editor with preview, reset, and JSON import/export."""

    def __new__(cls, figure, visual_id: str, style: VisualStyle, theme_name: str,
                parent=None, on_preview=None):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
            QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
            QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QToolButton,
            QVBoxLayout, QWidget,
        )
        from pyCamSet.gui.colour_picker import ColourPicker
        from pyCamSet.gui.figure_fonts import available_fonts, resolve_figure_font

        class _Dialog(QDialog):
            def __init__(self):
                super().__init__(parent)
                self.setWindowTitle("Figure style")
                self.setModal(True)
                self.visual_id = visual_id
                self.original = VisualStyle(**asdict(style))
                self.current = VisualStyle(**asdict(style))
                self.original_artist_state = _capture_presentation_state(figure)
                self.original_override = _VISUAL_OVERRIDES.get(figure)
                self.theme_name = theme_name
                self.on_preview = on_preview
                self._preview_applied = False
                root = QVBoxLayout(self)
                content = QWidget(self)
                content_layout = QVBoxLayout(content)
                content_layout.setSpacing(10)

                def group(title, host=None):
                    box = QGroupBox(title, content)
                    box_form = QFormLayout(box)
                    box_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
                    (host or content_layout).addWidget(box)
                    return box, box_form

                # ---- Suggested appearance (first: the quickest start) ----
                _preset_box, preset_form = group("Suggested appearance")
                self.preset = QComboBox()
                self.preset.addItem("Custom", None)
                for key, preset in PRESETS.items():
                    self.preset.addItem(preset["label"], key)
                self.preset.setCurrentIndex(max(0, self.preset.findData(style.suggested_preset)))
                self.preset.setToolTip("A starting point you can edit; not a claim of journal compliance.")
                preset_form.addRow("Preset:", self.preset)

                # ---- Text and Figure, side by side as in the optical-mapping GUI ----
                pair = QHBoxLayout()
                content_layout.addLayout(pair)
                _text_box, text_form = group("Text", pair)
                self.font = QComboBox()
                # Only open-source families with a recorded licence that can
                # render here; an older style's other family shows as the
                # approved family that replaces it.
                self.font.addItems(["Sans Serif"] + list(available_fonts()))
                self.font.setCurrentText(resolve_figure_font(style.font_family) or "Sans Serif")
                self.font.setToolTip("Open-source typefaces with a recorded licence; "
                                     "DejaVu families are bundled and always available.")
                text_form.addRow("Typeface:", self.font)
                self.font_weight = QComboBox()
                self.font_weight.addItems(["Theme default", "Normal", "Bold"])
                self.font_weight.setCurrentIndex({None: 0, "normal": 1, "bold": 2}[style.font_weight])
                text_form.addRow("Weight:", self.font_weight)
                self.font_size = _number(QDoubleSpinBox, style.font_size, 10, 1, 48)
                self.font_size.setSuffix(" pt")
                text_form.addRow("Size:", self.font_size)
                self.text_colour = ColourPicker(style.text_colour)
                text_form.addRow("Text colour:", self.text_colour)
                self.title_colour = ColourPicker(style.title_colour)
                text_form.addRow("Title colour:", self.title_colour)

                # ---- Figure ----
                _figure_box, figure_form = group("Figure", pair)
                self.figure_background = ColourPicker(style.figure_background)
                figure_form.addRow("Figure background:", self.figure_background)
                self.axes_background = ColourPicker(style.axes_background)
                figure_form.addRow("Plot background:", self.axes_background)
                self.line_width = _number(QDoubleSpinBox, style.line_width, 1.5, 0.2, 12)
                self.line_width.setToolTip("Width of every line; a row in the table below can override one line.")
                figure_form.addRow("All line widths:", self.line_width)
                self.marker_size = _number(QDoubleSpinBox, style.marker_size, 6, 1, 24)
                figure_form.addRow("All line markers:", self.marker_size)
                self.grid = QCheckBox("Override grid")
                self.grid.setChecked(style.grid_visible is not None)
                self.grid_value = QCheckBox("Show grid")
                self.grid_value.setChecked(bool(style.grid_visible))
                figure_form.addRow(self.grid, self.grid_value)
                self.legend = QCheckBox("Override legend")
                self.legend.setChecked(style.legend_visible is not None)
                self.legend_value = QCheckBox("Show legend")
                self.legend_value.setChecked(bool(style.legend_visible))
                figure_form.addRow(self.legend, self.legend_value)

                # ---- Every element: lines, thresholds, bars, points ----
                self.elements = [row for row in styleable_elements(figure) if row["kind"] != "detections"]
                self.element_table = QTableWidget(len(self.elements), 7, content)
                self.element_table.setObjectName("elementTable")
                self.element_table.setHorizontalHeaderLabels(
                    ["Element", "Colour", "Width", "Line style", "Shape", "Fill", "Size"])
                self.element_table.verticalHeader().setVisible(False)
                self.element_table.setAccessibleName("Figure elements and their styles")
                header = self.element_table.horizontalHeader()
                header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
                header.setMinimumSectionSize(60)
                for column in range(1, 7):
                    header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
                self.element_table.setWordWrap(False)
                self._element_controls = []
                for row, element in enumerate(self.elements):
                    self._element_controls.append(self._build_element_row(row, element))
                self.element_table.resizeColumnsToContents()
                self.element_table.resizeRowsToContents()
                rows_height = sum(self.element_table.rowHeight(row) for row in range(len(self.elements)))
                # Show every row without an inner scroll bar, up to about eight rows.
                table_height = min(self.element_table.horizontalHeader().sizeHint().height()
                                   + rows_height + 6, 360)
                self.element_table.setMinimumHeight(table_height)
                self.element_table.setMaximumHeight(table_height)
                elements_box = QGroupBox("Lines, thresholds, bars and points", content)
                elements_layout = QVBoxLayout(elements_box)
                if self.elements:
                    hint = QLabel("“Original” keeps the look the figure was drawn with. "
                                  "Thresholds and reference lines are listed separately from data, "
                                  "so a preset never recolours them unless you do.")
                    hint.setWordWrap(True)
                    hint.setProperty("textRole", "muted")
                    elements_layout.addWidget(hint)
                    elements_layout.addWidget(self.element_table)
                else:
                    self.element_table.hide()
                    elements_layout.addWidget(QLabel("This figure has no lines, bars or points to restyle."))
                content_layout.addWidget(elements_box)

                # ---- Detection markers (only where the figure has them) ----
                self.detections_box, detection_form = group("Detection markers")
                # Short values: controls keep their natural width, not the dialog's.
                detection_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
                self.overlay_colour = ColourPicker(style.overlay_colour)
                self.overlay_colour.setAccessibleName("Detection marker colour")
                detection_form.addRow("Colour:", self.overlay_colour)
                self.overlay_marker = QComboBox()
                self.overlay_marker.addItem("Theme default", "")
                for label, marker in MARKER_SHAPES:
                    self.overlay_marker.addItem(label, marker)
                self.overlay_marker.setCurrentIndex(max(0, self.overlay_marker.findData(style.overlay_marker or "")))
                self.overlay_marker.setAccessibleName("Detection marker shape")
                detection_form.addRow("Shape:", self.overlay_marker)
                self.overlay_fill = QComboBox()
                self.overlay_fill.addItems(["Theme default", "Filled", "Hollow (outline only)"])
                self.overlay_fill.setCurrentIndex({None: 0, True: 1, False: 2}[style.overlay_filled])
                self.overlay_fill.setAccessibleName("Detection marker fill")
                self.overlay_fill.setToolTip("Hollow markers leave the corner visible inside them. "
                                             "Plus, cross and point shapes are always outlines.")
                detection_form.addRow("Fill:", self.overlay_fill)
                for control in (self.overlay_marker, self.overlay_fill):
                    control.setMinimumWidth(220)
                self.overlay_size = _number(QDoubleSpinBox, style.overlay_size, 10, 1, 24)
                self.overlay_size.setAccessibleName("Detection marker size")
                detection_form.addRow("Size:", self.overlay_size)
                self.overlay_line_width = _number(QDoubleSpinBox, style.overlay_line_width, 0.4, 0.2, 12)
                self.overlay_line_width.setAccessibleName("Detection marker edge width")
                detection_form.addRow("Outline width:", self.overlay_line_width)
                self.overlay_opacity = _number(QDoubleSpinBox, style.overlay_opacity, 1.0, 0.0, 1.0)
                self.overlay_opacity.setDecimals(2)
                self.overlay_opacity.setSingleStep(0.05)
                self.overlay_opacity.setAccessibleName("Detection marker opacity")
                detection_form.addRow("Opacity:", self.overlay_opacity)
                self.detections_box.setVisible(any(row["kind"] == "detections"
                                                   for row in styleable_elements(figure)))

                # ---- Advanced: exact colours and rarely used controls ----
                self.advanced_toggle = QToolButton(content)
                self.advanced_toggle.setObjectName("sectionToggle")
                self.advanced_toggle.setCheckable(True)
                self.advanced_toggle.setText("▶  Advanced")
                self.advanced_toggle.setAccessibleName("Advanced settings")
                self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
                content_layout.addWidget(self.advanced_toggle)
                self.advanced = QGroupBox("", content)
                advanced_form = QFormLayout(self.advanced)
                advanced_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
                content_layout.addWidget(self.advanced)
                advanced_note = QLabel("Every colour control also takes an exact value: open it "
                                       "and choose “Custom colour… (exact RGB)”.")
                advanced_note.setWordWrap(True)
                advanced_note.setProperty("textRole", "muted")
                advanced_form.addRow(advanced_note)
                self.tick_colour = ColourPicker(style.tick_colour)
                advanced_form.addRow("Tick label colour:", self.tick_colour)
                self.axes_colour = ColourPicker(style.axes_colour)
                advanced_form.addRow("Axis label colour:", self.axes_colour)
                self.legend_colour = ColourPicker(style.legend_colour)
                advanced_form.addRow("Legend text colour:", self.legend_colour)
                self.overlay_edge_colour = ColourPicker(style.overlay_edge_colour)
                self.overlay_edge_colour.setAccessibleName("Detection marker edge colour")
                advanced_form.addRow("Detection outline colour:", self.overlay_edge_colour)
                self.overlay_line_style = QComboBox()
                self.overlay_line_style.addItems(["Theme default", "Solid", "Dashed", "Dash-dot", "Dotted"])
                self.overlay_line_style.setCurrentIndex({None: 0, "-": 1, "--": 2, "-.": 3, ":": 4}[style.overlay_line_style])
                self.overlay_line_style.setAccessibleName("Detection marker edge line style")
                advanced_form.addRow("Detection outline style:", self.overlay_line_style)
                self.series_id = QComboBox()
                self.series_id.addItem("No series override", "")
                for axes_index, axes in enumerate(figure.axes):
                    for line in axes.lines:
                        label = line.get_label()
                        if label and not label.startswith("_"):
                            stable_id = line.get_gid() or f"line:{axes_index}:{label}"
                            self.series_id.addItem(f"{label} [{stable_id}]", stable_id)
                advanced_form.addRow("Series by ID:", self.series_id)
                self.series_colour = ColourPicker("")
                self.series_width = _number(QDoubleSpinBox, None, 1.5, 0.2, 12)
                self.series_dash = QComboBox()
                self.series_dash.addItems(["Theme default", "Solid", "Dashed", "Dash-dot", "Dotted"])
                self.series_marker = QComboBox()
                self.series_marker.addItems(["Theme default", "None", "Circle", "Square", "Triangle up",
                                             "Triangle down", "Diamond", "Plus", "Cross", "Point"])
                if self.series_id.count() < 2:
                    self.series_id.setEnabled(False)
                    self.series_colour.setEnabled(False)
                advanced_form.addRow("Series colour:", self.series_colour)
                advanced_form.addRow("Series line width:", self.series_width)
                advanced_form.addRow("Series dash:", self.series_dash)
                advanced_form.addRow("Series marker:", self.series_marker)
                self.colormap = QComboBox()
                self.colormap.addItems(["Theme default", "viridis", "plasma", "inferno", "magma",
                                        "cividis", "coolwarm", "RdBu_r"])
                can_recolour = any((image.get_gid() or "").startswith("style:colormap:")
                                   for axes in figure.axes for image in axes.images)
                self.colormap.setEnabled(can_recolour)
                self.colormap.setToolTip("Only explicitly opted-in scalar images can change palette; scientific ranges are unchanged.")
                advanced_form.addRow("Opted-in image palette:", self.colormap)
                scale = QCheckBox("Enable scale bar")
                scale.setEnabled(False)
                scale.setToolTip(scale_bar_unavailable())
                advanced_form.addRow("Scale bar:", scale)
                scale_note = QLabel("Numeric limits, units, colourbar labels, heatmap palettes and signed/error-map encodings are fixed by the scientific producer. Range controls stay unavailable unless a producer explicitly supplies non-semantic presentation limits.")
                scale_note.setWordWrap(True)
                scale_note.setAccessibleName("Scientific colour-scale limits are protected")
                advanced_form.addRow(scale_note)
                citation = QLabel(SUGGESTED_PRESET_CITATIONS)
                citation.setProperty("sourceRevision", SUGGESTED_PRESET_REGISTRY_VERSION)
                citation.setWordWrap(True)
                citation.setAccessibleName("Suggested figure-style sources and limitations")
                advanced_form.addRow(citation)
                note = QLabel("Scale bars require a known pixel-to-world transform and unit. "
                              "Raw-image overlays remain unmodified.")
                note.setWordWrap(True)
                advanced_form.addRow(note)
                self.advanced.setVisible(False)

                def toggle_advanced(checked):
                    self.advanced.setVisible(checked)
                    self.advanced_toggle.setText(("▼" if checked else "▶") + "  Advanced")
                self.advanced_toggle.toggled.connect(toggle_advanced)
                content_layout.addStretch(1)
                self.preset.currentIndexChanged.connect(self._apply_suggested_preset)
                scroll = QScrollArea(self)
                scroll.setWidgetResizable(True)
                scroll.setWidget(content)
                root.addWidget(scroll, stretch=1)
                # Keep Cancel and the save controls outside the scrolling area.
                # Bound the modal to the usable screen even at high DPI.
                screen = self.screen()
                if screen is not None:
                    available = screen.availableGeometry()
                    self.setMaximumHeight(max(320, available.height() - 80))
                    self.resize(min(980, available.width() - 60),
                                min(available.height() - 80, 860))
                row = QHBoxLayout()
                root.addLayout(row)
                save = QPushButton("Save JSON…")
                load = QPushButton("Load JSON…")
                reset = QPushButton("Reset to theme")
                reset.setToolTip("Clear this figure's settings; it then follows the saved "
                                 "default style, or the GUI theme when there is none.")
                row.addWidget(save)
                row.addWidget(load)
                row.addWidget(reset)
                default_row = QHBoxLayout()
                root.addLayout(default_row)
                self.save_default = QPushButton("Save as default for all figures")
                self.save_default.setToolTip(
                    "Use these settings for every figure that has no style of its own, "
                    "including figures opened later. Per-series colours and colour maps "
                    "stay with this figure.")
                self.clear_default = QPushButton("Clear default")
                self.clear_default.setToolTip("Figures without their own style follow the GUI theme again.")
                self.default_status = QLabel("")
                self.default_status.setWordWrap(True)
                self.default_status.setProperty("textRole", "muted")
                default_row.addWidget(self.save_default)
                default_row.addWidget(self.clear_default)
                default_row.addWidget(self.default_status, stretch=1)
                self.save_default.clicked.connect(self._save_as_default)
                self.clear_default.clicked.connect(self._clear_default)
                self._show_default_status()
                buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                           QDialogButtonBox.StandardButton.Cancel)
                root.addWidget(buttons)
                for widget in (self.font, self.font_weight, self.font_size, self.line_width, self.marker_size,
                               self.text_colour, self.title_colour, self.tick_colour, self.axes_colour,
                               self.legend_colour, self.figure_background, self.axes_background,
                               self.overlay_size, self.overlay_marker, self.overlay_colour, self.overlay_edge_colour,
                               self.overlay_line_width, self.overlay_line_style, self.overlay_opacity,
                               self.overlay_fill,
                               self.series_id, self.series_colour, self.series_width,
                               self.series_dash, self.series_marker, self.colormap, self.preset,
                               self.grid, self.grid_value,
                               self.legend, self.legend_value,
                               *(control for controls in self._element_controls
                                 for control in controls.values())):
                    signal = getattr(widget, "currentTextChanged", None) or getattr(widget, "valueChanged", None) \
                        or getattr(widget, "textChanged", None) or getattr(widget, "toggled", None)
                    if signal is not None:
                        signal.connect(self._preview)
                self.series_id.currentIndexChanged.connect(self._load_series_colour)
                reset.clicked.connect(self._reset)
                save.clicked.connect(self._save)
                load.clicked.connect(self._load)
                buttons.accepted.connect(self.accept)
                buttons.rejected.connect(self.reject)

            def _build_element_row(self, row, element):
                """One table row: only the controls that apply to this kind of element."""
                allowed = ELEMENT_KEYS[element["kind"]]
                saved = dict(style.series_styles.get(element["id"], {}))
                if "colour" not in saved and element["id"] in style.series_colours:
                    saved["colour"] = style.series_colours[element["id"]]
                name = QTableWidgetItem(element["name"])
                name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
                name.setToolTip({"line": "Data line", "reference": "Threshold or reference line",
                                 "bars": "Bar group", "points": "Scatter points"}[element["kind"]])
                self.element_table.setItem(row, 0, name)
                colour = ColourPicker(saved.get("colour"), default_label="Original")
                width = QDoubleSpinBox()
                width.setRange(0.0, 12.0)
                width.setSingleStep(0.25)
                width.setSpecialValueText("Original")
                width.setValue(saved.get("line_width", 0.0))
                line_style = QComboBox()
                line_style.addItem("Original", None)
                for label, code in LINE_STYLES:
                    line_style.addItem(label, code)
                line_style.setCurrentIndex(max(0, line_style.findData(saved.get("line_style"))))
                shape = QComboBox()
                shape.addItem("Original", None)
                for label, code in MARKER_SHAPES:
                    shape.addItem(label, code)
                shape.setCurrentIndex(max(0, shape.findData(saved.get("marker") or None)))
                fill = QComboBox()
                fill.addItems(["Original", "Filled", "Hollow"])
                fill.setCurrentIndex({None: 0, True: 1, False: 2}[saved.get("filled")])
                size = QDoubleSpinBox()
                size.setRange(0.0, 48.0)
                size.setSpecialValueText("Original")
                size.setValue(saved.get("size", 0.0))
                controls = {"colour": colour, "line_width": width, "line_style": line_style,
                            "marker": shape, "filled": fill, "size": size}
                for column, key in enumerate(("colour", "line_width", "line_style", "marker", "filled", "size"), start=1):
                    control = controls[key]
                    control.setEnabled(key in allowed)
                    control.setAccessibleName(f"{element['name']} {key.replace('_', ' ')}")
                    self.element_table.setCellWidget(row, column, control)
                return controls

            def _element_settings(self, controls):
                """The non-original settings of one table row."""
                settings = {}
                if controls["colour"].isEnabled() and controls["colour"].text():
                    settings["colour"] = controls["colour"].text()
                if controls["line_width"].isEnabled() and controls["line_width"].value() > 0:
                    settings["line_width"] = max(0.2, controls["line_width"].value())
                if controls["line_style"].isEnabled() and controls["line_style"].currentData():
                    settings["line_style"] = controls["line_style"].currentData()
                if controls["marker"].isEnabled() and controls["marker"].currentData():
                    settings["marker"] = controls["marker"].currentData()
                if controls["filled"].isEnabled() and controls["filled"].currentIndex():
                    settings["filled"] = controls["filled"].currentIndex() == 1
                if controls["size"].isEnabled() and controls["size"].value() > 0:
                    settings["size"] = max(1.0, controls["size"].value())
                return settings

            def _set_element_rows(self, candidate):
                """Show *candidate*'s element settings in the table, without previewing."""
                for element, controls in zip(self.elements, self._element_controls):
                    saved = dict(candidate.series_styles.get(element["id"], {}))
                    if "colour" not in saved and element["id"] in candidate.series_colours:
                        saved["colour"] = candidate.series_colours[element["id"]]
                    for control in controls.values():
                        control.blockSignals(True)
                    controls["colour"].setText(saved.get("colour", ""))
                    controls["line_width"].setValue(saved.get("line_width", 0.0))
                    controls["line_style"].setCurrentIndex(max(0, controls["line_style"].findData(saved.get("line_style"))))
                    controls["marker"].setCurrentIndex(max(0, controls["marker"].findData(saved.get("marker") or None)))
                    controls["filled"].setCurrentIndex({None: 0, True: 1, False: 2}[saved.get("filled")])
                    controls["size"].setValue(saved.get("size", 0.0))
                    for control in controls.values():
                        control.blockSignals(False)

            def _read(self):
                series_colours = dict(self.current.series_colours)
                series_styles = {key: dict(value) for key, value in self.current.series_styles.items()}
                for element, controls in zip(self.elements, self._element_controls):
                    settings = self._element_settings(controls)
                    legacy = series_colours.get(element["id"])
                    if legacy is not None and settings.get("colour") == legacy:
                        # An unchanged colour from an older file stays where it was.
                        settings.pop("colour")
                    elif legacy is not None:
                        series_colours.pop(element["id"], None)
                    if settings:
                        series_styles[element["id"]] = settings
                    else:
                        series_styles.pop(element["id"], None)
                selected_id = self.series_id.currentData()
                if selected_id:
                    properties = {}
                    if self.series_width.value() != 1.5:
                        properties["line_width"] = self.series_width.value()
                    if self.series_dash.currentIndex():
                        properties["line_style"] = ["-", "-", "--", "-.", ":"][self.series_dash.currentIndex()]
                    if self.series_marker.currentIndex():
                        properties["marker"] = ["", "", "o", "s", "^", "v", "D", "+", "x", "."][self.series_marker.currentIndex()]
                    colour = self.series_colour.text().strip()
                    if colour:
                        if series_colours.get(str(selected_id)) == colour:
                            pass  # Keep legacy colour-only overrides in their v1 field.
                        else:
                            series_colours.pop(str(selected_id), None)
                            properties["colour"] = colour
                    else:
                        series_colours.pop(str(selected_id), None)
                    if properties:
                        series_styles[str(selected_id)] = properties
                    else:
                        series_styles.pop(str(selected_id), None)
                return VisualStyle(
                    font_family=(self.font.currentText()
                                 if self.font.currentText() != "Sans Serif" else None),
                    font_weight=(None if self.font_weight.currentIndex() == 0
                                 else ("normal" if self.font_weight.currentIndex() == 1 else "bold")),
                    font_size=self.font_size.value() if self.font_size.value() != 10 else None,
                    text_colour=_optional(self.text_colour.text()),
                    title_colour=_optional(self.title_colour.text()),
                    tick_colour=_optional(self.tick_colour.text()),
                    axes_colour=_optional(self.axes_colour.text()),
                    legend_colour=_optional(self.legend_colour.text()),
                    figure_background=_optional(self.figure_background.text()),
                    axes_background=_optional(self.axes_background.text()),
                    line_width=self.line_width.value() if self.line_width.value() != 1.5 else None,
                    marker_size=self.marker_size.value() if self.marker_size.value() != 6 else None,
                    grid_visible=self.grid_value.isChecked() if self.grid.isChecked() else None,
                    legend_visible=self.legend_value.isChecked() if self.legend.isChecked() else None,
                    overlay_size=(self.overlay_size.value()
                                  if self.overlay_size.value() != 10 else None),
                    overlay_marker=self.overlay_marker.currentData() or None,
                    overlay_colour=_optional(self.overlay_colour.text()),
                    overlay_edge_colour=_optional(self.overlay_edge_colour.text()),
                    overlay_line_width=(self.overlay_line_width.value()
                                        if self.overlay_line_width.value() != 0.4 else None),
                    overlay_line_style=(None if self.overlay_line_style.currentIndex() == 0
                                        else ("-", "-", "--", "-.", ":")[self.overlay_line_style.currentIndex()]),
                    overlay_opacity=(self.overlay_opacity.value()
                                     if self.overlay_opacity.value() != 1.0 else None),
                    overlay_filled={0: None, 1: True, 2: False}[self.overlay_fill.currentIndex()],
                    series_colours=series_colours, series_styles=series_styles,
                    colormap=(self.colormap.currentText() if self.colormap.currentIndex() else None),
                    suggested_preset=self.preset.currentData(),
                    suggested_preset_registry=(
                        None if self.preset.currentData() is None
                        else SUGGESTED_PRESET_REGISTRY_VERSION),
                    # The palette belongs to the chosen preset; Custom has none.
                    series_palette=(list(PRESETS[self.preset.currentData()]["series_palette"])
                                    if self.preset.currentData() is not None else None))

            def _load_series_colour(self, *_):
                stable_id = self.series_id.currentData()
                settings = self.current.series_styles.get(stable_id, {})
                self.series_colour.setText(settings.get("colour")
                                            or self.current.series_colours.get(stable_id, ""))
                self.series_width.setValue(settings.get("line_width", 1.5))
                self.series_dash.setCurrentIndex({"-": 1, "--": 2, "-.": 3, ":": 4}.get(settings.get("line_style"), 0))
                self.series_marker.setCurrentIndex({"o": 2, "s": 3, "^": 4, "v": 5, "D": 6,
                                                    "+": 7, "x": 8, ".": 9}.get(settings.get("marker"), 0))

            def _apply_suggested_preset(self, index):
                """Fill the controls from a shared preset: a starting point, never compliance."""
                key = self.preset.itemData(index)
                if key is None:
                    return
                preset = PRESETS[key]
                self.font.setCurrentText(resolve_figure_font(preset["font_family"]))
                self.font_weight.setCurrentIndex({"normal": 1, "bold": 2}[preset["font_weight"]])
                self.font_size.setValue(preset["font_size"])
                self.line_width.setValue(preset["line_width"])
                self.marker_size.setValue(preset["marker_size"])
                self.text_colour.setText(preset["text_colour"])
                self.title_colour.setText(preset["title_colour"])
                self.figure_background.setText(preset["figure_background"])
                self.axes_background.setText(preset["axes_background"])
                self.tick_colour.clear()
                self.axes_colour.clear()
                self.legend_colour.clear()

            def _preview(self, *_, force=False):
                try:
                    candidate = self._read()
                    candidate.validate()
                    if candidate == self.current and not force:
                        return
                    self.current = candidate
                    apply_visual_style(figure, candidate, self.theme_name)
                    self._preview_applied = True
                    if self.on_preview:
                        self.on_preview(self.current)
                except ValueError:
                    pass

            def _reset(self):
                self.current = VisualStyle()
                self.font.setCurrentText("Sans Serif")
                self.font_weight.setCurrentIndex(0)
                self.font_size.setValue(10)
                self.line_width.setValue(1.5)
                self.marker_size.setValue(6)
                self.text_colour.clear()
                self.title_colour.clear()
                self.tick_colour.clear()
                self.axes_colour.clear()
                self.legend_colour.clear()
                self.figure_background.clear()
                self.axes_background.clear()
                self.overlay_size.setValue(10)
                self.overlay_marker.setCurrentIndex(0)
                self.overlay_colour.clear()
                self.overlay_edge_colour.clear()
                self.overlay_line_width.setValue(0.4)
                self.overlay_line_style.setCurrentIndex(0)
                self.overlay_opacity.setValue(1.0)
                self.series_id.setCurrentIndex(0)
                self.series_colour.clear()
                self.series_width.setValue(1.5)
                self.series_dash.setCurrentIndex(0)
                self.series_marker.setCurrentIndex(0)
                self.colormap.setCurrentIndex(0)
                self.preset.setCurrentIndex(0)
                self.grid.setChecked(False)
                self.legend.setChecked(False)
                self.overlay_fill.setCurrentIndex(0)
                self._set_element_rows(self.current)
                self._preview(force=True)

            def _show_default_status(self):
                from pyCamSet.gui.preferences import config_directory
                exists = default_style_path(config_directory()).is_file()
                self.clear_default.setEnabled(exists)
                self.default_status.setText(
                    "A default style is saved." if exists
                    else "No default style: figures follow the GUI theme.")

            def _save_as_default(self):
                from PySide6.QtWidgets import QMessageBox
                from pyCamSet.gui.preferences import config_directory
                try:
                    candidate = self._read()
                    candidate.validate()
                    save_default_style(config_directory(), candidate)
                except (OSError, ValueError) as exc:
                    QMessageBox.warning(self, "Default style not saved",
                                        f"The default style could not be saved.\n\nTechnical detail: {exc}")
                    return
                self._show_default_status()
                self.default_status.setText("Saved: figures without their own style now use it.")

            def _clear_default(self):
                from PySide6.QtWidgets import QMessageBox
                from pyCamSet.gui.preferences import config_directory
                try:
                    clear_default_style(config_directory())
                except OSError as exc:
                    QMessageBox.warning(self, "Default style not cleared",
                                        f"The default style could not be removed.\n\nTechnical detail: {exc}")
                    return
                self._show_default_status()

            def _save(self):
                from PySide6.QtWidgets import QMessageBox
                path, _ = QFileDialog.getSaveFileName(self, "Save visual style", "visual-style.json",
                                                      "JSON files (*.json)")
                if path:
                    try:
                        _validate_user_style_filename(path)
                        self.current = self._read()
                        Path(path).write_text(style_to_json(self.current, visual_id), encoding="utf-8")
                    except (OSError, ValueError) as exc:
                        QMessageBox.warning(self, "Style not saved",
                                            f"The style could not be saved.\n\nTechnical detail: {exc}")

            def _load(self):
                from PySide6.QtWidgets import QMessageBox
                path, _ = QFileDialog.getOpenFileName(self, "Load visual style", "",
                                                      "JSON files (*.json)")
                if not path:
                    return
                try:
                    candidate = style_from_json(Path(path).read_text(encoding="utf-8"), visual_id)
                    if candidate.colormap and not self.colormap.isEnabled():
                        raise ValueError("This visual does not explicitly support colormap changes")
                except (OSError, ValueError) as exc:
                    QMessageBox.warning(self, "Style not loaded",
                                        f"The style file was rejected.\n\nTechnical detail: {exc}")
                    return
                self.current = candidate
                controls = (self.font, self.font_weight, self.font_size, self.line_width, self.marker_size,
                            self.text_colour, self.title_colour, self.tick_colour, self.axes_colour,
                            self.legend_colour, self.figure_background, self.axes_background,
                            self.overlay_size, self.overlay_marker, self.overlay_colour, self.overlay_edge_colour,
                            self.overlay_line_width, self.overlay_line_style, self.overlay_opacity,
                            self.series_id, self.series_colour, self.series_width, self.series_dash,
                            self.series_marker, self.colormap, self.preset, self.grid, self.grid_value,
                            self.legend, self.legend_value)
                blocked = [control.blockSignals(True) for control in controls]
                try:
                    self.font.setCurrentText(resolve_figure_font(candidate.font_family) or "Sans Serif")
                    self.font_weight.setCurrentIndex({None: 0, "normal": 1, "bold": 2}[candidate.font_weight])
                    self.font_size.setValue(candidate.font_size or 10)
                    self.line_width.setValue(candidate.line_width or 1.5)
                    self.marker_size.setValue(candidate.marker_size or 6)
                    self.text_colour.setText(candidate.text_colour or "")
                    self.title_colour.setText(candidate.title_colour or "")
                    self.tick_colour.setText(candidate.tick_colour or "")
                    self.axes_colour.setText(candidate.axes_colour or "")
                    self.legend_colour.setText(candidate.legend_colour or "")
                    self.figure_background.setText(candidate.figure_background or "")
                    self.axes_background.setText(candidate.axes_background or "")
                    self.overlay_size.setValue(candidate.overlay_size if candidate.overlay_size is not None else 10)
                    self.overlay_marker.setCurrentIndex(max(0, self.overlay_marker.findData(candidate.overlay_marker or "")))
                    self.overlay_colour.setText(candidate.overlay_colour or "")
                    self.overlay_edge_colour.setText(candidate.overlay_edge_colour or "")
                    self.overlay_line_width.setValue(candidate.overlay_line_width or 0.4)
                    self.overlay_line_style.setCurrentIndex({None: 0, "-": 1, "--": 2, "-.": 3, ":": 4}[candidate.overlay_line_style])
                    self.overlay_opacity.setValue(candidate.overlay_opacity if candidate.overlay_opacity is not None else 1.0)
                    self.series_id.setCurrentIndex(0)
                    self.series_colour.clear()
                    self.series_width.setValue(1.5)
                    self.series_dash.setCurrentIndex(0)
                    self.series_marker.setCurrentIndex(0)
                    for index in range(self.series_id.count()):
                        stable_id = self.series_id.itemData(index)
                        if stable_id in candidate.series_colours or stable_id in candidate.series_styles:
                            self.series_id.setCurrentIndex(index)
                            self._load_series_colour()
                            break
                    self.grid.setChecked(candidate.grid_visible is not None)
                    self.grid_value.setChecked(bool(candidate.grid_visible))
                    self.legend.setChecked(candidate.legend_visible is not None)
                    self.legend_value.setChecked(bool(candidate.legend_visible))
                    self.colormap.setCurrentIndex(self.colormap.findText(candidate.colormap or "Theme default"))
                    self.preset.setCurrentIndex(max(0, self.preset.findData(candidate.suggested_preset)))
                    self.overlay_fill.setCurrentIndex({None: 0, True: 1, False: 2}[candidate.overlay_filled])
                    self._set_element_rows(candidate)
                finally:
                    for control, was_blocked in zip(controls, blocked):
                        control.blockSignals(was_blocked)
                self._preview(force=True)

            def exec(self):
                result = super().exec()
                if result != QDialog.DialogCode.Accepted and self._preview_applied:
                    _restore_presentation_state(figure, self.original_artist_state)
                    if self.original_override is None:
                        _VISUAL_OVERRIDES.pop(figure, None)
                    else:
                        _VISUAL_OVERRIDES[figure] = self.original_override
                    if self.on_preview:
                        self.on_preview(self.original)
                else:
                    self.current = self._read()
                return result

        def _number(widget_type, value, default, minimum, maximum):
            widget = widget_type()
            widget.setRange(minimum, maximum)
            widget.setValue(value if value is not None else default)
            return widget

        def _colour(_widget_type, value):
            return ColourPicker(value)

        def _optional(value):
            return value.strip() or None

        return _Dialog()
