# The graphical workflow

`pycamset` launches the graphical calibration workflow. It is part of the
default install, and it is a front end to
[the workflow package](../internals/workflow.md): the GUI reads settings off
its forms, hands them to the same phase functions a script would call, and
writes the same runs into the same workspace. Nothing is calculated here that
cannot be calculated without a display.

```bash
pycamset
```

If it will not start, see
[Troubleshooting](../troubleshooting.md#pycamset-will-not-start).

---

## The window

![The pyCamSet main window, showing the Phase 0 Data Input tab.](../assets/gui/phase-0-data-input-light.png#only-light)
![The pyCamSet main window, showing the Phase 0 Data Input tab.](../assets/gui/phase-0-data-input-dark.png#only-dark)

The **File**, **Edit**, and **Settings** menus sit at the top left. File also
offers **Create Target…**; it remains available as a prominent button above
the tabs. Edit contains the informational-tooltip toggle. Settings contains
the terminal-output toggle and a **Theme** submenu. Every item is an ordinary
menu entry that can be clicked (on macOS, too, where the menu bar is the
system's own) and stays in step with the setting it controls.

**Run** on a phase can be pressed once: it reads **Running…** and stays
disabled until that run finishes, fails or is cancelled, and a rerun started
from the phase's diagnostics holds it the same way.

The Settings theme selector offers Light, Dark, and Sepia. The selected theme
sets application chrome and the neutral chrome of GUI-managed Matplotlib
figures (figure/axes backgrounds, labels, ticks, borders, and legend frames).
Managed PyVista assessment views inherit the selected theme for their default
background; an explicit saved per-visual background overrides that default.
Neither theme nor presentation styles recolour plotted data, overlays, or
scientific colormaps. Open3D retains its native interactive view and does not
support these managed 3D style controls.

The window's own title bar follows the theme as well. On Windows 11 it takes
the theme's exact colours, so it reads as one strip with the menu bar; on
Windows 10 and macOS it switches between the system's light and dark frames.
On Linux the desktop's window manager draws the frame and decides.

These presentation choices are remembered for the current operating-system
user: the theme remains in Qt's existing per-user settings, while tooltip and
terminal visibility, export presets, and saved visual styles live under Qt's
pyCamSet application-config directory. On Windows this is under the user's
local application data (`%LOCALAPPDATA%/pyCamSet/pyCamSet`); on macOS use
`~/Library/Preferences/pyCamSet`, and on Linux use
`$XDG_CONFIG_HOME/pyCamSet` or `~/.config/pyCamSet`. The JSON file is
`preferences.json`; reset those
choices by closing pyCamSet and deleting that file. The `visual-styles`
directory contains the existing per-visual style files and can be reset
separately. Neither location is a project or run directory. A malformed
preferences file is left intact and defaults are used; the original is copied
to `preferences.json.corrupt` if a later preference edit replaces it.
An explicit `PYCAMSET_CONFIG_DIR` environment override takes precedence over
Qt's standard location, for isolated or portable setups.

Editable parameter inputs on Phase 0–4, Export Calibration, Optimisation and
Detection Cost are saved in a separate `parameters.json` in that same per-user
directory when the main window closes. They return on the next launch, including
the Phase 3 `max_nfev` value (factory default: **1000**), target settings and
detector options. Use **Reset Parameters to Default** on any tab to restore only
that tab's factory input values; the reset is saved immediately. Run selectors,
diagnostic selections, camera selection and target settings inherited from a
selected Phase 1/2 run are not restored as global parameters: select the run or
confirm the image folder again before processing. Generated calibration results
are never restored as parameters. A malformed settings file is preserved as
`parameters.json.corrupt` before a new settings file replaces it. No parameter
settings are written into a reconstruction workspace or the source repository
by default (unless `PYCAMSET_CONFIG_DIR` explicitly points there).

The persistent action row below each phase's parameters keeps its run,
diagnostics, and available continue/assessment actions in reach while expanded
settings scroll independently. At smaller window sizes, use the parameter
panel's scroll area to reach the remaining settings.

### Action icon meanings

pyCamSet shares its figure-action icons with the lab's optical-mapping (OM)
GUI, so the same action looks the same in both tools. Each is a compact
28 × 28 px button; hover it for the action's name, which is also its
accessible name for screen readers and keyboard users.

| pyCamSet action | Icon | Code point | OM source action | pyCamSet behaviour |
| --- | --- | --- | --- | --- |
| Figure and detection-overlay style | ⚙ | U+2699 | `src/ui/figure_style.py:1817` (“⚙ Options”) | Opens the visual-style controls. |
| Save figure or montage as PNG | 📷 | U+1F4F7 | `src/ui/figure_export.py:88` (camera button) | Saves the rendered figure or montage as PNG. |
| Save figure data as CSV | 📊 | U+1F4CA | `src/ui/figure_export.py:941` (table button) | Saves source-backed numeric data; disabled when no rows are available. |

The icons are Unicode characters drawn by the operating system's own symbol or
emoji font; pyCamSet ships no icon image or font for them. On a system with no
such font (common on minimal Linux installs), pyCamSet draws an equivalent
gear, camera or bar-chart outline instead, so a button never shows an empty
box.

No icon is assigned to playback or ROI controls: these are not equivalent to
the cited OM figure actions. SVG/PDF exports keep text labels because they are
vector-file formats, not the OM chart-data action.

The following controls are global to the workflow:

**Create Target…**
:   Opens the target generator, described in [Making a target](#making-a-target).
    It is a dialog rather than a tab because a target is drawn once, printed,
    and then used for months of runs — it used to sit ahead of Phase 0, so every
    session opened on the one step almost no session takes.

**Enable Informational Windows**
:   Turns hover tooltips on and off. Most controls explain themselves and say
    what a sensible value looks like. This toggle is in **Edit**.

**Show Terminal Output**
:   Shows or hides the terminal pane at the bottom of each phase tab. That pane
    is where a phase's output appears while it runs. This toggle is in
    **Settings**.

Behind the tab bar there are twelve tabs, of which eight are visible: the five
phases, Export Calibration, Optimisation, and Detection Cost. The other four
are diagnostics companions, one per phase, hidden until that phase has produced
a run.

In a narrow window the tab bar scrolls, and tabs at the end slide out of view
behind two small arrows. The **All tabs ▾** menu at the right-hand end of the
tab bar always lists every visible tab, with the open one ticked; choose a tab
there to jump to it.

---

## Working through the phases

### Phase 0 — Data Input

Points the workflow at a folder of images, one sub-folder per camera, and
checks that what it finds can be calibrated at all. It reports three
diagnostics:

- **D0.1** the number of camera sub-folders found
- **D0.2** the number of images in each
- **D0.3** whether those counts are consistent across cameras

The workspace defaults to `<f_loc>/.pycamset_workspace`, a directory beside the
images that holds every run. Recent folders are remembered between sessions.

### Phase 1 — Detection

![The Phase 1 Detection tab.](../assets/gui/phase-1-detection-light.png#only-light)
![The Phase 1 Detection tab.](../assets/gui/phase-1-detection-dark.png#only-dark)

When a run finishes, a card to the right of the controls shows two images with
their detections drawn on: the image with the most detections and the image
with the fewest (among those where the target was found at all), each named
with its camera, file and count. **See more in Diagnostics** opens the full
viewer. The markers follow the style saved for the diagnostics overlay.

Finds the calibration target in every image. The calibration target is
described here, and the detector's own options appear in a section named for
the backend in use — *Detection Options (aruco1)* above.

Phase 1 is also where the **Detector** is chosen, because which library reads
the markers is a property of the detection rather than of the printed target.
For ChArUco1 and ChArUco1 ccube targets, pick **ArUco 1 (OpenCV)** or
**ArUco 2 (aruco2)**. A ChArUco2 or ChArUco2 ccube target can only be read
with ArUco 2, so selecting it chooses ArUco 2 by itself and greys ArUco 1 out;
switching back restores the detector you last chose for a ChArUco1 or ChArUco1
ccube target. PuzzleBoard targets have their own detector and show no choice.
Detections are cached per detector, so a run with one never picks up the other's
cache.

Both detectors read the same printed ChArUco1/ChArUco1 ccube board, so this is
a detection-quality choice, not a printing one. In synthetic tests ArUco 1 (the
default) gave the lowest corner outlier counts; ArUco 2 found more corners
under blur, noise and strong tilt, with comparable camera pose accuracy, but
produced more corners over 3 px off and a worse lens-model fit on some cube
datasets. These are synthetic-test results — try both on your own images if
in doubt.

Leave **Cache detections** on unless there is a reason not to: with it off the
run is saved with its diagnostics but no detections, and the later phases have
nothing to pick up.

Diagnostics run to D1.7, and include the per-camera detection rate (D1.2),
board completeness (D1.3), a feature-count heatmap (D1.4), spatial coverage
(D1.6) and the minimum feature count (D1.7).

### Phase 2 — Intrinsics

After a run, the card beside the controls plots every image's reprojection
RMS for each camera, with a diamond for the camera's own RMS: which camera is
off, and whether a few images carry the error, at a glance.

Calibrates each camera independently, producing the camset the bundle
adjustment starts from. Its diagnostics cover per-camera RMS reprojection
(D2.1), the intrinsics and distortion themselves (D2.2, D2.3), the spread
across views (D2.5) and per-view error (D2.6, D2.7).

Phases 2 and 3 offer no detector: they read the detections of the Phase 1 run
they continue, so the target section shows that run's detector instead.

**Lens model** is set here, because this is the phase that builds the cameras.
Leave it on Pinhole for a conventional lens. Choose Telecentric when the optics
are, or the fit is badly conditioned: a telecentric lens images with parallel
rays, so magnification does not fall off with distance, and a pinhole model has
no way to express that except by driving the focal length towards infinity and
absorbing the rest into the distortion coefficients. The symptom is a camera
whose focal length is implausibly large for its sensor, whose `fx` and `fy`
disagree despite square pixels, and whose first distortion coefficient is far
from zero — together with a rig that will not hold still in the Phase 3
consistency report. Phase 3 needs no setting of its own; it reads the model
from the cameras Phase 2 produced.

### Phase 3 — Bundle Adjustment

![The Phase 3 Bundle Adjustment tab.](../assets/gui/phase-3-bundle-adjustment-light.png#only-light)
![The Phase 3 Bundle Adjustment tab.](../assets/gui/phase-3-bundle-adjustment-dark.png#only-dark)

After a run, the card beside the controls shows the camera poses, each camera
named with its mean reprojection error underneath. A telecentric camera has
only its direction solved, so a telecentric rig is drawn one unit back along
each camera's viewing direction, looking at the target; the card says so. The
D3.13 diagnostics plot draws the same figure.

Solves the whole camera set and the target poses together. The solver controls
are here — thread count, `max_nfev`, verbosity, outlier rejection, and which
camera and pose to hold fixed as the gauge.

**Camera Lockbox Prior** is the optional section that holds the cameras near a
known rig geometry. Its editor is a translation-only tool for repairing camera
centres non-destructively: it works on a derived camset rather than the
original, with trust flags, plane groups, undo and redo. It is marked
experimental.

Diagnostics run to D3.13, and are the ones to read when a calibration is
disappointing: initial and final euclidean error (D3.5, D3.6), the reduction
ratio between them (D3.7), solver status and evaluation count (D3.8, D3.9), the
parameter-to-observation ratio (D3.10), a residual scatter (D3.11), per-camera
mean reprojection (D3.12) and an extrinsic pose view (D3.13).

### Phase 4 — Self-Calibration

Lets the target's own points move as well, gauged against a fixed subset so the
solve cannot simply scale everything away. Useful when the printed target is
not quite the shape it was drawn as — which is
[stage 5 of a calibration](calibrate.md#stage-5-freeing-the-targets-geometry),
run for you.

Its diagnostics say how far the target moved (D4.7), how flat it stayed (D4.9),
and how the result compares against phase 3 (D4.4), which is the number that
says whether letting the shape move helped.

---

## Continue buttons, and when they turn red

Each phase ends with a green **Continue to Next Phase** button. After a run
finishes, that button is judged against the run it would carry: a phase can
finish successfully and still have produced nothing worth going on with — a
camera that saw the target in no image cannot be calibrated, and a solve that
ended at NaN has not solved anything.

When that happens the button turns red, the reasons are printed to the terminal
pane prefixed with `Cannot continue:`, and clicking it asks for confirmation
before proceeding. **The button is never disabled.** You are always allowed to
carry on; you are just told first, because the judgement is a heuristic and you
may know better.

## Diagnostics tabs

Each phase's **Diagnostics** button reveals its companion tab, which stays
hidden in the tab bar until then. A diagnostics tab compares *runs*, not just
the last one: it lists every saved run for that phase and pre-selects the three
most recent, so a change in settings can be read against what came before.

Each also shows an **Upstream Run Chain** for the selected run — the phase 2 it
started from, the phase 1 that fed that — so a result can be traced back to the
detections that produced it.

Managed Matplotlib figure cards keep a compact header: the camera icon saves a
PNG, the **▾** beside it holds **Save SVG**, **Save PDF** and the **Export size**,
the gear opens **Style…**, the chart icon saves a CSV when a source-backed numeric
adapter exists, and **Expand** opens a larger view. Export sizes are screen
(160 mm / 150 dpi), generic single-column (85 mm / 300 dpi), and generic
double-column (180 mm / 300 dpi). These are templates, not
claims of compliance with a named journal. CSV exports include JSON metadata comments
for source/run, data kind, axes and units where those are known; raster screenshots
are never converted into fabricated data.
The
style editor (⚙) previews every change on the figure as you make it. It is
arranged in groups:

- **Suggested appearance**: a starting preset (see below).
- **Text** and **Figure**: typeface, weight, size, text and title colours,
  figure and plot backgrounds, the width of all lines and the size of all line
  markers, and grid and legend visibility.
- **Lines, thresholds, bars and points**: one row per element of the figure,
  each with its own colour, width, line style, marker shape, fill (filled or
  hollow) and size. Only the controls that apply to an element are enabled:
  bars have no line style, for example. "Original" keeps the look the figure
  was drawn with. Thresholds and reference lines are separate rows from data,
  so a preset never recolours them unless you do.
- **Detection markers** (detection montages): colour, shape, fill, size,
  outline width and opacity. A hollow circle or square leaves the detected
  corner visible inside it; plus, cross and point shapes are outlines only.
- **Advanced** (collapsed): tick, axis-label and legend text colours, detection
  outline colour and style, a single-series editor addressed by ID, and the
  colour map of images that allow one.

Every colour control opens a palette of named colours — neutrals, the
colour-blind-safe Okabe–Ito set, and common journal colours — plus "Theme
default" and **Custom colour… (exact RGB)** for a precise value. The source
image and detection coordinates are never edited. Styles are kept per visual in the application
configuration directory, separately from scientific run artefacts, and can be
saved to or loaded from versioned JSON. Unknown or malformed style fields are
rejected rather than partially applied.

**Suggested appearance** offers the same six starting points as the lab's
optical-mapping GUI: Nature, Science, Cell and IEEE suggestions, **Presentation
(large)** (18 pt text and heavy lines for projection) and **Grayscale/minimal**
(black text and a grey series ramp for print). Each fills the typography, line,
marker, text and background controls, which stay editable, and recolours the
figure's data line series from its palette. Reference lines such as thresholds,
bars, scatter points and images keep their own colours, and a series given its
own colour keeps it.

**Save as default for all figures** makes the current settings the style of
every figure that has no style of its own, including figures opened later;
per-series colours and colour maps stay with the figure they were set on.
**Clear default** returns those figures to the GUI theme. **Reset to theme**
removes the figure's own style, after which it follows the saved default, or
the theme when there is none.

The typeface list holds only open-source families with a recorded licence (the
same list as the optical-mapping GUI). The bundled DejaVu families are always
available; Liberation, Noto, Open Sans, Roboto, Lato, Source Sans Pro, PT Sans,
PT Serif and Latin Modern Roman appear when installed. A style saved earlier
with another family is drawn with the approved family of the same kind, for
example DejaVu Serif in place of Times New Roman. A scale
bar remains unavailable unless a calibrated pixel-to-world transform and units
are supplied; no physical length is inferred. Error bars remain unavailable:
the GUI has no source-backed uncertainty producer or error-bar control, so
fail-closed behaviour for unavailable uncertainty cannot yet be
tested. The journal styles are suggestions, not journal-compliance
presets. The Science link cites the 2025 *Guide to Preparing Figures*; its PDF
was inspected in a 30 July 2026 archive snapshot because a direct publisher
fetch returned 403. The separate 2026 *Science Advances* guide is not the
source for this Science suggestion. This presentation editor applies
to Matplotlib cards and the Phase 1 detection montage; the montage has a frame PNG
and observed-coordinate CSV action. Phase 2 per-view plots and Phase 3 error,
residual and camera-pose plots are hosted in cards. See [visual export coverage](visual-export.md)
for the surface-by-surface traceability and remaining gaps. Managed PyVista
views also provide 3D background, point-size, camera-view, axes,
and error-legend controls. Save style and Load saved operate on a versioned
per-visual preference in the user configuration directory; Reset style removes
that override and returns to theme-default inheritance. The settings are
presentation-only and do not modify calibration coordinates, error scalars,
or run artefacts. Open3D controls that cannot be implemented are explicitly
unavailable; target printable SVG/PDF generation remains unchanged.

Phase 3 and Phase 4 diagnostics share an **Assess Calibration** page:

- **Figures** — the error distribution, per-camera coverage and accuracy vs
  precision plots draw in the page as soon as it is opened for a run (the
  calibration is assessed off the GUI thread). They are ordinary figure cards,
  with style, PNG/SVG/PDF and source-backed CSV, and they reflow from three
  columns to one as the window narrows.
- **3D views** — the reconstructed points with the cameras, and in the
  target's own frame. They use more memory, so they wait for **Visualise
  Calibration**. *In this tab (PyVista)* embeds both interactive views
  (through `pyvistaqt`); *Separate window (PyVista)* and *Separate window
  (Open3D)* open a viewer of their own. **3D style ▾** holds the background,
  point size, view, axes and legend, which redraw the embedded views live.
- **Export ▾** saves the figures (PNG, SVG, PDF and CSV), the 3D views as a
  PNG, or 3D geometry (glTF, OBJ, PLY), at the chosen export size.

On a dark background the camera frustums and the 3D titles and legend are drawn
in light grey rather than black, so they stay visible.

## Making a target

![The Create Target dialog.](../assets/gui/create-target-light.png#only-light)
![The Create Target dialog.](../assets/gui/create-target-dark.png#only-dark)

**Create Target…** generates a printable ChArUco1, ChArUco1 ccube, ChArUco2,
ChArUco2 ccube, PuzzleBoard or PuzzleBoardCube. Both halves of the form are
built from what the selected target declares about itself — the arguments that
decide what it is, and the options that decide how it is drawn — so the form and
the validation follow the target rather than being written out per target.

The dialog asks for no detector: a ChArUco1 or ChArUco1 ccube prints identically
for ArUco 1 and ArUco 2, so the same printed board can be read with either, and
the other targets have only one detector. The detector is chosen in Phase 1
instead.

The labels are the names the GUI shows; saved settings and scripts use the class
names, which are unchanged: ChArUco1 is `ChArUco`, ChArUco1 ccube is `Ccube`,
ChArUco2 is `ChArUco2`, and ChArUco2 ccube is `Ccube2`.

**Visualise Target** opens the target in a window of its own, so it can be
compared against the form that drew it. **Save Target** writes the SVG or PDF.
Print at 100% scale: the export carries the true dimensions, and scaling to fit
the page silently invalidates the target.

## Exporting

**Export Calibration** writes a selected Phase 3 or Phase 4 calibration to
COLMAP `sparse/0` format, for use by photogrammetry and neural reconstruction
tools. It is a wrapper around
[`camset_to_colmap`][pyCamSet.utils.saving.camset_to_colmap], which a script
can call directly.

## Optimisation

![The Optimisation tab.](../assets/gui/optimisation-light.png#only-light)
![The Optimisation tab.](../assets/gui/optimisation-dark.png#only-dark)

Sweeps the detector's settings, scores the calibrations that result, and retains
the best. The settings offered are the ones the selected target's detector says
it can be swept over. A study runs detections of its own, so this tab keeps a
**Detector** choice, which behaves as it does in Phase 1. The work runs on a
background thread, so the window stays responsive.

This tab needs Optuna, which is optional. Without it the tab still appears, but
**Start** is disabled with a tooltip saying why:

```bash
pip install "pyCamSet[optimisation]"
```

---

## Regenerating these screenshots

The images on this page are produced by a script, so they can be refreshed when
the GUI changes:

```bash
python scripts/capture_gui_screenshots.py
```

It renders each tab offscreen in the application's own Light and Dark themes,
with a throwaway settings directory, so the output depends neither on the
desktop theme nor on the theme, folders or parameters of whoever ran it. On
Windows, where the offscreen renderer has no fonts, run it with
`QT_QPA_PLATFORM=windows`; nothing is shown on screen either way.
