"""Interactive scenes and real terminal output for the documentation examples.

Every example in the docs is executed while the site is built, and whatever it
produced is shown underneath the code -- a PyVista scene as a turnable vtk.js
frame, a matplotlib figure as an SVG, and anything the run logged as the block
of terminal text it actually was.  Either way the example keeps the plain
``cams.plot()`` or ``calibrate_cameras(...)`` spelling a reader would type: a
scene is caught through PyVista's own gallery mechanism, where
``PYVISTA_BUILDING_GALLERY`` leaves the serialised scene on ``last_vtksz``
after ``show()``, and a figure is simply one still open when the block ends.

Each scene is written out as a page carrying its own copy of the vtk.js
viewer, straight into the built site, so nothing generated lands in ``docs/``
and a frame has nothing to fetch once it has loaded.  A scene carrying point
labels is the exception: that vtk.js build has no label engine, so such a
scene is rendered to a PNG instead and shown as an image.
"""

import os

# Both must be set before pyvista is imported.
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("PYVISTA_BUILDING_GALLERY", "true")

import logging
from io import BytesIO, StringIO
from pathlib import Path

import matplotlib

# No display while the site builds, and the examples draw before they are shown.
matplotlib.use("Agg")

import markdown_exec
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from PIL import Image
from markupsafe import Markup, escape
from mkdocs.utils import get_relative_url
from pyvista.plotting.plotter import _ALL_PLOTTERS
from trame_vtk.tools.vtksz2html import write_html

# pyvista reads the flag once, as it is imported, and keeps a shown plotter
# alive only while it is set.  Had anything imported pyvista before this module,
# the setdefault above would have come too late and every scene would be dropped.
if not pv.BUILDING_GALLERY:
    raise RuntimeError("pyvista was imported before PYVISTA_BUILDING_GALLERY was set")

# A scene is addressed relative to the site root, which is a different number of
# levels up on every page; on_page_content substitutes the right one.
ROOT = "%%SCENE_ROOT%%"

FRAME = (
    '<iframe class="scene" data-src="{root}{name}"'
    ' style="width:100%;aspect-ratio:{aspect};border:1px solid'
    ' var(--md-default-fg-color--lightest);border-radius:.2rem"></iframe>'
)

# Each frame is a separate vtk.js viewer -- a megabyte of javascript and a WebGL
# context of its own -- and a browser keeps only a handful of contexts alive.  A
# page with several would drop the ones it could not keep, which is why a scene
# would arrive late or blank.  So a frame is handed its source when it comes near
# the viewport and has it taken away again when it leaves: only the scenes the
# reader is actually looking at are ever live.
FRAME_LOADER = """
<script>
(() => {
  const frames = document.querySelectorAll('iframe.scene[data-src]');
  if (!frames.length) return;
  const watch = new IntersectionObserver((entries) => {
    for (const {target, isIntersecting} of entries) {
      if (isIntersecting === !!target.src) continue;
      if (isIntersecting) target.src = target.dataset.src;
      else target.removeAttribute('src');
    }
  }, {rootMargin: '400px'});
  frames.forEach((frame) => watch.observe(frame));
})();
</script>
"""

# The frame takes the shape of the plotter, so an example that wants a wide scene
# asks for a wide window.  PyVista's own default means "no preference", and gets
# the shape these pages read best at.
DEFAULT_ASPECT = "16/10"

IMAGE = (
    '<img class="scene" loading="lazy" src="{root}{name}"'
    ' style="width:100%;border:1px solid'
    ' var(--md-default-fg-color--lightest);border-radius:.2rem">'
)

# A plot is drawn at a size, and an SVG carries it; stretching every one to the
# column would redraw a single-panel figure at twice the scale of a four-panel
# one, and its text with it.  So a figure is shown at the size it was drawn,
# centred, and only shrinks when the column is narrower than that.
FIGURE = (
    '<img class="figure" loading="lazy" src="{root}{name}"'
    ' style="max-width:100%;display:block;margin:0 auto;border:1px solid'
    ' var(--md-default-fg-color--lightest);border-radius:.2rem">'
)

scenes: dict[str, bytes] = {}

# The viewer's vtk.js build has no renderPointsAsSpheres and no
# renderLinesAsTubes, so a scene drawn with either arrives as flat square dots
# and hairlines.  It does honour geometry, line width and opacity, which is what
# these two put back.
MIN_LINE_OPACITY = 0.01
POINT_RADIUS = 0.004  # of the scene diagonal, per sqrt(point in pixels)
# Glyphing turns one point into a sphere's worth of them, so the resolution steps
# down as the cloud grows and the whole scene stays affordable.  A sphere of
# resolution r carries r * (r - 2) + 2 points; below the coarsest one the markers
# are left flat, which is what vtk.js would have drawn anyway.
GLYPH_POINT_BUDGET = 60_000
SPHERE_RESOLUTIONS = (10, 8, 6)


def aspect(plotter):
    """The frame shape this scene asked for.

    Read before ``show()``, which takes the render window down with it and
    leaves ``window_size`` unreadable.
    """
    width, height = plotter.window_size
    if (width, height) == tuple(pv.global_theme.window_size):
        return DEFAULT_ASPECT
    return f"{width}/{height}"


def sphere_resolution(n_points):
    """The roundest sphere this many markers can afford, or None for flat."""
    for r in SPHERE_RESOLUTIONS:
        if n_points * (r * (r - 2) + 2) <= GLYPH_POINT_BUDGET:
            return r
    return None


def web_safe(plotter):
    """Redraw what vtk.js cannot, in terms it can.

    Detected features become real sphere geometry, and a camera frustum keeps
    its lines but at an opacity that survives a 2px hairline.
    """
    for renderer in plotter.renderers:
        lo, hi = np.array(renderer.bounds).reshape(3, 2).T
        diagonal = float(np.linalg.norm(hi - lo)) or 1.0
        for actor in list(renderer.actors.values()):
            prop = getattr(actor, "prop", None)
            dataset = getattr(getattr(actor, "mapper", None), "dataset", None)
            if prop is None or dataset is None:
                continue
            if prop.render_points_as_spheres and sphere_resolution(dataset.n_points):
                res = sphere_resolution(dataset.n_points)
                radius = diagonal * POINT_RADIUS * prop.point_size**0.5
                sphere = pv.Sphere(radius, theta_resolution=res, phi_resolution=res)
                actor.mapper.dataset = dataset.glyph(
                    geom=sphere, scale=False, orient=False)
                prop.render_points_as_spheres = False
                # `add_points` draws with style='points', which would render the
                # glyphed spheres as their own vertices -- flat dots again.
                prop.style = "surface"
            elif prop.render_lines_as_tubes or str(prop.style).lower() == "wireframe":
                prop.opacity = max(prop.opacity, MIN_LINE_OPACITY)
                prop.line_width = max(prop.line_width, 2)
                prop.render_lines_as_tubes = False


def labelled(plotter):
    """Whether the scene carries labels, which only a real render can draw.

    ``add_point_labels`` places its text through a label-placement mapper, and
    the viewer's vtk.js build has nothing that drives one, so such a scene
    arrives with the labels simply missing.  It is the mapper that identifies
    them: subplot borders and scalar bars are 2-D actors too, and both survive
    the export.
    """
    return any(isinstance(actor.GetMapper(), pv._vtk.vtkLabelPlacementMapper)
               for renderer in plotter.renderers
               for actor in renderer.actors.values()
               if isinstance(actor, pv._vtk.vtkActor2D))


def screenshot(plotter):
    """The scene as PNG bytes, rendered by VTK itself."""
    buffer = BytesIO()
    Image.fromarray(plotter.screenshot(return_img=True)).save(buffer, "PNG")
    return buffer.getvalue()


# The scene is serialised inside show(), so the fixup has to happen just before
# it -- and the screenshot before the fixup, which rewrites geometry VTK draws
# well.
_show = pv.Plotter.show


def show(self, *args, **kwargs):
    self._doc_png = screenshot(self) if labelled(self) else None
    self._doc_aspect = aspect(self)
    web_safe(self)
    result = _show(self, *args, **kwargs)
    # pyvista exports the scene behind a suppressed ImportError, so an
    # unreachable trame component leaves `last_vtksz` unset and the scene out of
    # the page -- silently, and a --strict build still passes with nothing in it.
    if self._doc_png is None and self.last_vtksz is None:
        raise RuntimeError(
            "pyvista exported no scene; its trame component is unreachable, which"
            " is `pip install trame-pyvista` and the trame versions it pins"
        )
    # What marks a plotter as this block's, rather than the export doing it: a
    # labelled scene is a screenshot and has no export to be recognised by.
    self._doc_shown = True
    return result


pv.Plotter.show = show


# What the block that is currently running has logged.
logged = StringIO()

# The rendered log, styled as the text block markdown-exec produces for a
# block's printed output, so logged and printed output read the same.
TERMINAL = '<div class="language-text highlight"><pre><code>{text}</code></pre></div>'


class TerminalLog(logging.Handler):
    """Collect pyCamSet's records for the block that is running.

    The calibration summary, the detection report and the intrinsics report all
    reach a user through ``logger.info`` rather than ``print``, so capturing
    them is the only way the docs can show what a calibration actually says.
    markdown-exec captures a block's output by handing the code its own
    ``print`` rather than by redirecting ``sys.stdout``, so a record has no way
    of reaching that buffer -- hence a buffer of our own, emptied into the page
    once the block has run.

    Installing this also settles what the records look like: ``setup_logging``
    leaves a logger that already has a handler alone, so pyCamSet's coloured
    handler never goes on and no ANSI reaches the HTML.
    """

    def emit(self, record):
        logged.write(self.format(record) + "\n")


def terminal():
    """Whatever the block that just ran logged, as a block of terminal text."""
    text = logged.getvalue()
    logged.seek(0)
    logged.truncate()
    return TERMINAL.format(text=escape(text.strip())) if text.strip() else ""


def capture(page):
    """Draw out every scene the block that just ran has shown."""
    frames = []
    for key in [k for k, p in _ALL_PLOTTERS.items() if getattr(p, "_doc_shown", False)]:
        plotter = _ALL_PLOTTERS.pop(key)
        png = plotter._doc_png
        name = f"{page}-{len(scenes):02d}" + (".png" if png else ".html")
        scenes[name] = png if png else viewer(plotter.last_vtksz)
        frames.append(IMAGE.format(root=ROOT, name=name) if png else
                      FRAME.format(root=ROOT, name=name, aspect=plotter._doc_aspect))
        plotter.close()
    return "".join(frames)


def viewer(vtksz):
    """A scene as a page that carries its own viewer.

    ``write_html`` inlines the vtk.js viewer and the scene itself, base64, into
    one document.  The alternative -- one shared viewer fetching a ``.vtksz``
    beside it -- is a request that has to survive whatever serves the site, and
    on GitHub Pages it does not: the frame comes up empty.  A scene that is
    already whole when it loads has nothing left to go wrong.
    """
    document = StringIO()
    write_html(vtksz, document)
    return document.getvalue().encode("utf-8")


def figures(page):
    """Draw out every matplotlib figure the block that just ran has left open.

    Vector, because these are line plots read at whatever width the page is,
    and they carry their own white ground so a transparent SVG cannot pick up
    the theme's.
    """
    frames = []
    for number in plt.get_fignums():
        figure = plt.figure(number)
        buffer = BytesIO()
        figure.savefig(buffer, format="svg", bbox_inches="tight")
        name = f"{page}-{len(scenes):02d}.svg"
        scenes[name] = buffer.getvalue()
        frames.append(FIGURE.format(root=ROOT, name=name))
        plt.close(figure)
    return "".join(frames)


def formatter(source, language, css_class, options, md, **kwargs):
    """markdown-exec's python formatter, plus whatever the code drew and said."""
    # Output mode is markdown-exec's own: unset renders captured stdout as
    # markdown, `result="text"` as a block of terminal text.  A block that
    # writes anything a reader is meant to read as output -- printed, or logged,
    # which is how a calibration reports itself -- asks for it in the fence.
    html = markdown_exec.formatter(source, language, css_class, options, md, **kwargs)
    # Markup escapes whatever is concatenated onto it, and the frames are html.
    page = options.get("session") or "scene"
    return Markup(str(html) + terminal() + capture(page) + figures(page))


def on_config(config):
    # The fence is registered here rather than in mkdocs.yml because a
    # `!!python/name:` tag is resolved as the YAML is read, before this file --
    # and so the formatter above -- can be imported.
    superfences = config.mdx_configs.setdefault("pymdownx.superfences", {})
    fences = superfences.setdefault("custom_fences", [])
    # `mkdocs serve` re-runs this on every rebuild, and the fence is registered once.
    if not any(f["format"] is formatter for f in fences):
        fences.append(
            {
                "name": "python",
                "class": "python",
                "validator": markdown_exec.validator,
                "format": formatter,
            }
        )

    logger = logging.getLogger("pyCamSet")
    if not any(isinstance(h, TerminalLog) for h in logger.handlers):
        handler = TerminalLog()
        # Bare messages: the reports are already laid out in columns, and a
        # timestamp and level in front of every line would wrap them.
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return config


def on_pre_build(**kwargs):
    scenes.clear()


def on_page_content(html, page, **kwargs):
    html = html.replace(ROOT, get_relative_url("scenes/", page.file.url))
    return html + FRAME_LOADER if 'class="scene" data-src' in html else html


def on_post_build(config, **kwargs):
    out = Path(config.site_dir, "scenes")
    out.mkdir(parents=True, exist_ok=True)
    for name, data in scenes.items():
        (out / name).write_bytes(data)
