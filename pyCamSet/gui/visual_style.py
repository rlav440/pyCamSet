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
VERSION = 5
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
            if set(values) - {"line_width", "line_style", "marker", "colour"}:
                raise ValueError("unsupported per-series style control")
            if "line_width" in values and (not isinstance(values["line_width"], (int, float))
                    or isinstance(values["line_width"], bool) or not 0.2 <= values["line_width"] <= 12):
                raise ValueError("series line_width must be between 0.2 and 12")
            if values.get("line_style", "-") not in {"-", "--", "-.", ":"}:
                raise ValueError("unsupported series line_style")
            if values.get("marker", "") not in {"", "o", "s", "^", "v", "D", "+", "x", "."}:
                raise ValueError("unsupported series marker")
            if "colour" in values:
                _validate_colour(values["colour"], f"series_styles[{series_id!r}].colour")
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
    if document["schema"] != SCHEMA or type(document["version"]) is not int or document["version"] not in {1, 2, 3, 4, VERSION}:
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
    if set(values) != set(VisualStyle.__dataclass_fields__):
        raise ValueError("Style has missing or unknown keys")
    style = VisualStyle(**values)
    style.validate()
    return style


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
            series_id = line.get_gid()
            if series_id is None and line.get_label() and not line.get_label().startswith("_"):
                series_id = f"line:{axes_index}:{line.get_label()}"
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
        # Detection overlays opt in through a stable gid and are styled without
        # touching the underlying image or detection coordinate arrays.
        for collection in axes.collections:
            gid = collection.get_gid() or ""
            if gid.startswith("detection-overlay:"):
                baseline = overlay_baselines[collection]
                if hasattr(collection, "set_sizes"):
                    collection.set_sizes([style.overlay_size ** 2] if style.overlay_size is not None
                                         else baseline["sizes"])
                if hasattr(collection, "set_facecolor"):
                    collection.set_facecolor(style.overlay_colour or tokens["accent"])
                if hasattr(collection, "set_edgecolor"):
                    collection.set_edgecolor(style.overlay_edge_colour or tokens["border_strong"])
                if hasattr(collection, "set_paths"):
                    if style.overlay_marker is None:
                        collection.set_paths(baseline["paths"])
                    else:
                        from matplotlib.markers import MarkerStyle
                        marker = MarkerStyle(style.overlay_marker)
                        collection.set_paths([marker.get_path().transformed(marker.get_transform())])
                if hasattr(collection, "set_linewidths"):
                    collection.set_linewidths([style.overlay_line_width]
                                              if style.overlay_line_width is not None
                                              else baseline["linewidths"])
                if hasattr(collection, "set_linestyle"):
                    collection.set_linestyles(style.overlay_line_style or baseline["linestyles"])
                collection.set_alpha(style.overlay_opacity if style.overlay_opacity is not None
                                     else baseline["alpha"])
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
                       line.get_linestyle(), line.get_marker())
                      for line in axes.lines],
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
                            if (collection.get_gid() or "").startswith("detection-overlay:")],
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
        for line, width, marker_size, colour, linestyle, marker in axes_state["lines"]:
            line.set_linewidth(width)
            line.set_markersize(marker_size)
            line.set_color(colour)
            line.set_linestyle(linestyle)
            line.set_marker(marker)
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
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
            QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
            QScrollArea, QVBoxLayout, QWidget,
        )
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
                form = QFormLayout()
                content_layout.addLayout(form)
                self.font = QComboBox()
                # Only open-source families with a recorded licence that can
                # render here; an older style's other family shows as the
                # approved family that replaces it.
                self.font.addItems(["Sans Serif"] + list(available_fonts()))
                self.font.setCurrentText(resolve_figure_font(style.font_family) or "Sans Serif")
                self.font.setToolTip("Open-source typefaces with a recorded licence; "
                                     "DejaVu families are bundled and always available.")
                form.addRow("Typeface:", self.font)
                self.font_weight = QComboBox()
                self.font_weight.addItems(["Theme default", "Normal", "Bold"])
                self.font_weight.setCurrentIndex({None: 0, "normal": 1, "bold": 2}[style.font_weight])
                form.addRow("Font weight:", self.font_weight)
                self.font_size = _number(QDoubleSpinBox, style.font_size, 10, 1, 48)
                form.addRow("Typography size (pt):", self.font_size)
                self.line_width = _number(QDoubleSpinBox, style.line_width, 1.5, 0.2, 12)
                form.addRow("Line width:", self.line_width)
                self.marker_size = _number(QDoubleSpinBox, style.marker_size, 6, 1, 24)
                form.addRow("Marker size:", self.marker_size)
                self.text_colour = _colour(QLineEdit, style.text_colour)
                form.addRow("General text colour (#RRGGBB):", self.text_colour)
                self.title_colour = _colour(QLineEdit, style.title_colour)
                form.addRow("Title colour:", self.title_colour)
                self.tick_colour = _colour(QLineEdit, style.tick_colour)
                form.addRow("Tick colour:", self.tick_colour)
                self.axes_colour = _colour(QLineEdit, style.axes_colour)
                form.addRow("Axes-label colour:", self.axes_colour)
                self.legend_colour = _colour(QLineEdit, style.legend_colour)
                form.addRow("Legend text colour:", self.legend_colour)
                self.figure_background = _colour(QLineEdit, style.figure_background)
                form.addRow("Figure background:", self.figure_background)
                self.axes_background = _colour(QLineEdit, style.axes_background)
                form.addRow("Axes background:", self.axes_background)
                self.overlay_size = _number(QDoubleSpinBox, style.overlay_size, 10, 1, 24)
                self.overlay_size.setAccessibleName("Detection marker size")
                form.addRow("Detection marker size:", self.overlay_size)
                self.overlay_marker = QComboBox()
                self.overlay_marker.addItem("Theme default", "")
                for label, marker in (("Circle", "o"), ("Square", "s"), ("Triangle up", "^"),
                                      ("Triangle down", "v"), ("Diamond", "D"),
                                      ("Plus", "+"), ("Cross", "x"), ("Point", ".")):
                    self.overlay_marker.addItem(label, marker)
                self.overlay_marker.setCurrentIndex(max(0, self.overlay_marker.findData(style.overlay_marker or "")))
                self.overlay_marker.setAccessibleName("Detection marker shape")
                form.addRow("Detection marker shape:", self.overlay_marker)
                self.overlay_colour = _colour(QLineEdit, style.overlay_colour)
                self.overlay_colour.setAccessibleName("Detection marker fill colour")
                form.addRow("Detection marker colour:", self.overlay_colour)
                self.overlay_edge_colour = _colour(QLineEdit, style.overlay_edge_colour)
                self.overlay_edge_colour.setAccessibleName("Detection marker edge colour")
                form.addRow("Detection marker edge colour:", self.overlay_edge_colour)
                self.overlay_line_width = _number(QDoubleSpinBox, style.overlay_line_width, 0.4, 0.2, 12)
                self.overlay_line_width.setAccessibleName("Detection marker edge width")
                form.addRow("Detection marker edge width:", self.overlay_line_width)
                self.overlay_line_style = QComboBox()
                self.overlay_line_style.addItems(["Theme default", "Solid", "Dashed", "Dash-dot", "Dotted"])
                self.overlay_line_style.setCurrentIndex({None: 0, "-": 1, "--": 2, "-.": 3, ":": 4}[style.overlay_line_style])
                self.overlay_line_style.setAccessibleName("Detection marker edge line style")
                form.addRow("Detection marker edge style:", self.overlay_line_style)
                self.overlay_opacity = _number(QDoubleSpinBox, style.overlay_opacity, 1.0, 0.0, 1.0)
                self.overlay_opacity.setDecimals(3)
                self.overlay_opacity.setSingleStep(0.05)
                self.overlay_opacity.setAccessibleName("Detection marker opacity")
                form.addRow("Detection marker opacity:", self.overlay_opacity)
                self.series_id = QComboBox()
                self.series_id.addItem("No series override", "")
                for axes_index, axes in enumerate(figure.axes):
                    for line in axes.lines:
                        label = line.get_label()
                        if label and not label.startswith("_"):
                            stable_id = line.get_gid() or f"line:{axes_index}:{label}"
                            self.series_id.addItem(f"{label} [{stable_id}]", stable_id)
                form.addRow("Series ID:", self.series_id)
                self.series_colour = _colour(QLineEdit, "")
                self.series_width = _number(QDoubleSpinBox, None, 1.5, 0.2, 12)
                self.series_dash = QComboBox()
                self.series_dash.addItems(["Theme default", "Solid", "Dashed", "Dash-dot", "Dotted"])
                self.series_marker = QComboBox()
                self.series_marker.addItems(["Theme default", "None", "Circle", "Square", "Triangle up",
                                             "Triangle down", "Diamond", "Plus", "Cross", "Point"])
                if self.series_id.count() < 2:
                    self.series_id.setEnabled(False)
                    self.series_colour.setEnabled(False)
                form.addRow("Series colour (#RRGGBB):", self.series_colour)
                form.addRow("Series line width:", self.series_width)
                form.addRow("Series dash:", self.series_dash)
                form.addRow("Series marker:", self.series_marker)
                self.colormap = QComboBox()
                self.colormap.addItems(["Theme default", "viridis", "plasma", "inferno", "magma",
                                        "cividis", "coolwarm", "RdBu_r"])
                can_recolour = any((image.get_gid() or "").startswith("style:colormap:")
                                   for axes in figure.axes for image in axes.images)
                self.colormap.setEnabled(can_recolour)
                self.colormap.setToolTip("Only explicitly opted-in scalar images can change palette; scientific ranges are unchanged.")
                form.addRow("Opted-in scalar image palette:", self.colormap)
                scale_note = QLabel("Numeric limits, units, colourbar labels, heatmap palettes and signed/error-map encodings are fixed by the scientific producer. Range controls stay unavailable unless a producer explicitly supplies non-semantic presentation limits.")
                scale_note.setWordWrap(True)
                scale_note.setAccessibleName("Scientific colour-scale limits are protected")
                content_layout.addWidget(scale_note)
                self.preset = QComboBox()
                self.preset.addItem("Custom", None)
                for key, preset in PRESETS.items():
                    self.preset.addItem(preset["label"], key)
                self.preset.setCurrentIndex(max(0, self.preset.findData(style.suggested_preset)))
                form.addRow("Suggested appearance:", self.preset)
                citation = QLabel(SUGGESTED_PRESET_CITATIONS)
                citation.setProperty("sourceRevision", SUGGESTED_PRESET_REGISTRY_VERSION)
                citation.setWordWrap(True)
                citation.setAccessibleName("Suggested figure-style sources and limitations")
                content_layout.addWidget(citation)
                self.preset.currentIndexChanged.connect(self._apply_suggested_preset)
                self.grid = QCheckBox("Override grid visibility")
                self.grid.setChecked(style.grid_visible is not None)
                self.grid_value = QCheckBox("Show grid")
                self.grid_value.setChecked(bool(style.grid_visible))
                form.addRow(self.grid, self.grid_value)
                self.legend = QCheckBox("Override legend visibility")
                self.legend.setChecked(style.legend_visible is not None)
                self.legend_value = QCheckBox("Show legend")
                self.legend_value.setChecked(bool(style.legend_visible))
                form.addRow(self.legend, self.legend_value)
                scale = QCheckBox("Enable scale bar")
                scale.setEnabled(False)
                scale.setToolTip(scale_bar_unavailable())
                form.addRow("Scale bar:", scale)
                note = QLabel("Scale bars require a known pixel-to-world transform and unit. "
                              "Raw-image overlays remain unmodified.")
                note.setWordWrap(True)
                content_layout.addWidget(note)
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
                    self.resize(min(680, available.width() - 60),
                                min(available.height() - 80, 780))
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
                               self.series_id, self.series_colour, self.series_width,
                               self.series_dash, self.series_marker, self.colormap, self.preset,
                               self.grid, self.grid_value,
                               self.legend, self.legend_value):
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

            def _read(self):
                series_colours = dict(self.current.series_colours)
                series_styles = {key: dict(value) for key, value in self.current.series_styles.items()}
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

        def _colour(widget_type, value):
            widget = widget_type(value or "")
            widget.setPlaceholderText("Theme default")
            widget.setMaxLength(7)
            return widget

        def _optional(value):
            return value.strip() or None

        return _Dialog()
