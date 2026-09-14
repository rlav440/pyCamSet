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

Three things sit above the tabs and apply to all of them:

**Create Target…**
:   Opens the target generator, described in [Making a target](#making-a-target).
    It is a dialog rather than a tab because a target is drawn once, printed,
    and then used for months of runs — it used to sit ahead of Phase 0, so every
    session opened on the one step almost no session takes.

**Enable Informational Windows**
:   Turns hover tooltips on and off. Most controls explain themselves and say
    what a sensible value looks like.

**Show Terminal Output**
:   Shows or hides the terminal pane at the bottom of each phase tab. That pane
    is where a phase's output appears while it runs.

Behind the tab bar there are eleven tabs, of which seven are visible: the five
phases, Export Calibration, and Optimisation. The other four are diagnostics
companions, one per phase, hidden until that phase has produced a run.

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

Finds the calibration target in every image. The calibration target is
described here, and the detector's own options appear in a section named for
the backend in use — *Detection Options (aruco1)* above.

Leave **Cache detections** on unless there is a reason not to: with it off the
run is saved with its diagnostics but no detections, and the later phases have
nothing to pick up.

Diagnostics run to D1.7, and include the per-camera detection rate (D1.2),
board completeness (D1.3), a feature-count heatmap (D1.4), spatial coverage
(D1.6) and the minimum feature count (D1.7).

### Phase 2 — Intrinsics

Calibrates each camera independently, producing the camset the bundle
adjustment starts from. Its diagnostics cover per-camera RMS reprojection
(D2.1), the intrinsics and distortion themselves (D2.2, D2.3), the spread
across views (D2.5) and per-view error (D2.6, D2.7).

### Phase 3 — Bundle Adjustment

![The Phase 3 Bundle Adjustment tab.](../assets/gui/phase-3-bundle-adjustment-light.png#only-light)
![The Phase 3 Bundle Adjustment tab.](../assets/gui/phase-3-bundle-adjustment-dark.png#only-dark)

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
[self-calibration](self-calibration.md), run for you.

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

Figures are matplotlib cards with **Expand** and **Save PNG** buttons. Phase 3
and Phase 4 additionally offer **Assess Calibration**, which opens the full-size
reconstruction and residual views in native matplotlib and PyVista windows
rather than embedding them.

## Making a target

![The Create Target dialog.](../assets/gui/create-target-light.png#only-light)
![The Create Target dialog.](../assets/gui/create-target-dark.png#only-dark)

**Create Target…** generates a printable Ccube, ChArUco, PuzzleBoard or
PuzzleBoardCube. Both halves of the form are built from what the selected target
declares about itself — the arguments that decide what it is, and the options
that decide how it is drawn — so the form and the validation follow the target
rather than being written out per target.

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
it can be swept over. The work runs on a background thread, so the window stays
responsive.

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

It renders each tab offscreen with a pinned style and palette, in both light and
dark, so the output does not depend on the machine or the desktop theme it was
run on.
