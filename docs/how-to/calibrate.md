# Calibrating a camera set

[`calibrate_cameras`][pyCamSet.calibration.camera_calibrator.calibrate_cameras]
coordinates the whole job, from detection through to a finished `CameraSet`:

```python
cams = calibrate_cameras("my_rig", target, optimise_target=True)
```
This on call is a convinience wrapper around five sub-processes.
This page runs them one at a time on the Ccube corpus in this repository.

---

## The data it expects

Point it at a folder holding one sub-folder per camera:

```text
my_rig/
├── cam0/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── cam1/
└── cam2/
```

The sub-folder names become the camera names. Images that correspond to the
same instant should share a filename across the cameras.

## The target

Every stage below takes the target, because every stage needs to know what it is
looking at: the detector needs the markers, the per-camera calibration needs the
board layout, and the bundle adjustment needs the feature coordinates.

```python exec="true" source="above" session="calibrate"
from pathlib import Path

from cv2 import aruco

from pyCamSet import Ccube

target = Ccube(
    n_points=10,
    length=40,
    aruco_dict=aruco.DICT_6X6_1000,
    border_fraction=0.2,
)

f_loc = Path("tests/test_data/calibration_ccube")
```

## Stage 1 — Detection

Find the target in every image of every camera. This is the expensive half of a
calibration, and the only half that touches the images at all: everything after
it works from the detections.

```python exec="true" source="above" session="calibrate"
from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile

detections, camera_res = detect_datapoints_in_imfile(
    f_loc=f_loc,
    calibration_target=target,
    threads=1,
)
```

The result is cached beside the images as `detected_datapoints.pickle`, which is
what makes re-running a calibration cheap — the run above loads that cache
rather than re-detecting, but only when the cache was made for the same
target, cameras and image cap; otherwise it redetects and overwrites the
cache. Deleting it forces a fresh detection.

!!! note "Why `threads=1` here"

    Detection across several threads uses a process pool, and on any platform
    that spawns rather than forks, each worker re-imports the module it was
    started from. That is fine from a script guarded by
    `if __name__ == "__main__":`, and it is what you should use for real work —
    the default is two below your core count. It is not fine from inside a
    documentation build, whose main module is the build itself, so this page
    detects sequentially.

## Stage 2 — Reading the detections

The detection report says what was found before anything is solved. It is the
one report that catches a session that was never going to work: a camera that
saw the target in no image cannot be calibrated, and a board seen in only a
handful of images constrains almost nothing.

```python exec="true" source="above" session="calibrate"
from pyCamSet.calibration.camera_calibrator import (
    images_per_camera, validate_detections)

validate_detections(
    detections, target, image_counts=images_per_camera(f_loc))
```

`image_counts` is the denominator of the detection rate: a camera is measured
against the images its own folder held, not against what another camera
happened to have.
The number of images in which a camera detects a target is shown under 'detected',
while the average portion of the board detected is reported as 'complete'.
Completion is a nice to have, but if a camera has many failed detections this can cause
serious downstream issues.


## Stage 3 — One camera at a time

Each camera is calibrated on its own, by OpenCV, from the board observations it
made.

```python exec="true" source="above" session="calibrate"
from pyCamSet.calibration.camera_calibrator import (
    report_initial_calibration, run_initial_calibration)

initial_cams = run_initial_calibration(
    detections, target, camera_res, save=False)
initial_cams.set_resolutions_from_file(floc=f_loc)

report_initial_calibration(initial_cams, detections, target)
```

This initial report lets each camera best minimise its reprojection error.
For a non-planar geometry, like a cCube, this also treats the geometry as a collection of planes.
As a result, this error is much lower than achieved by a standard bundle adjustment.
However, it is a good metric of approximately how well a calibration perform once fully solved.


## Stage 4 — The multiview bundle adjustment

This is the standard multi-view calibration.


```python exec="true" source="above" session="calibrate"
from multiprocessing import cpu_count

from pyCamSet.calibration.camera_calibrator import run_stereo_calibration

cams = run_stereo_calibration(
    initial_cams, detections, target, save=False, threads=cpu_count())
```

Three blocks come out of this stage.

**Rig consistency** is drawn before the
solve: how much each camera's view of the target moves relative to the reference
camera's across the session. A rigid rig holds those near zero, and a camera that
does not is a camera that moved while the images were taken.
This is a canary for camera systems that were unstable/unsynced or, managed to generate
extremely varied estimates of the initial camera position.

**Per image initial reproj error** is the starting point after exploring an initial parameterisation.
For camera systems, we often do not have a fully shared pose, where every
camera detects the calibration at the same time. To initialise in this case, pyCamSet
runs a pathfinding algorithm over the connected camera and target poses.
It tries to find the initialisation that uses the fewest chained pose estimates.

The per-image reprojection error takes this estimate and runs the bundle adjustment loss as a sanity check.
If an image is orders of magnitudes worse than the others, it is unlikely for the calibration to solve.
A median absolute deviation threshold is used to automatically identify extreme outliers.
Note that the ~19 pixels of error is much worse than the sub 1 pixel initialisation.

**The calibration summary** reports on the solve.

- **Reprojection error** — `initial` is the mean Euclidean error before
  optimising and `final mean` after, so the gap between them is what the solve
  achieved. `median`, `rms`, `p95` and `max` describe the distribution: a
  median far below the mean means a small number of bad points are carrying the
  average.
- **The per-camera table** grades each camera's error by colour — under 0.1 px
  is exceptional, under 1 px good, under 5 px poor, above that bad. One camera
  far worse than the rest may warrant further investigation.
- **worst images** names the individual images contributing most of the error,
  which is where to look for a blurred frame or a target caught mid-move.
- **Flags** are the concerns worth a person's attention, raised by the run
  itself.

## Stage 5 — Freeing the target's geometry

Everything above treats the target as known: each feature sits exactly where the
target says it does. This stage runs the same observations a second time with
every feature coordinate free, so the difference between the target as drawn and
the target as printed, laminated, folded and handled comes out of the
reprojection error rather than being blamed on the cameras.

This is what `optimise_target=True` runs, and on this corpus it is not a
marginal effect: generally true for all cubic targets, as folding the net induces
significant fabrication error, and even 1 mm of error can result in large reprojection 
errors.

```python exec="true" source="above" session="calibrate"
from pyCamSet.calibration.camera_calibrator import run_self_calibration

free_cams = run_self_calibration(cams, detections, target, threads=cpu_count())
```

`cams` is left as the fixed-target solve rather than overwritten, because the
rest of this page compares the two.

`termination: max_iter reached` above is the default `max_nfev` of 100, not
convergence: freeing every feature coordinate makes a much larger problem than
the fixed-target one, and it is still descending when the cap stops it. Raise it
through `problem_options={"max_nfev": 400}` if the last fraction of a pixel
matters.

The cost is the gauge. Letting every point move leaves the problem
under-determined in seven degrees of freedom — the whole scene can translate,
rotate and scale without changing a single reprojection — and
[`SelfBundleHandler`][pyCamSet.optimisation.standard_bundle_handler.SelfBundleHandler]
breaks that symmetry by holding seven parameters of three non-colinear target
points. You do not have to choose them, but it is worth knowing it happened: the
result is only as correctly *scaled* as the points that were held, rather than
as the target as drawn.

How far the target moved is therefore worth as much attention as how far the
error fell. A large motion with a large error reduction is a target that was
genuinely misprinted. A large motion with only a small reduction is usually the
solve absorbing something else — a systematic detection bias, or an image set
too small to constrain the extra parameters. And if the error barely moves, the
target was not the limiting factor, and the fixed-target result is the one to
keep: it is gauged by a known object, which is a stronger claim about scale.

## The error distribution

The summary gives five numbers for a distribution; the distribution itself is
the residual the solver stopped at, carried on the finished set.
[`cluster_plot`][pyCamSet.utils.visualisation.cluster_plot] is what pyCamSet
draws it with — a 2D histogram of the per point error in *x* against the error
in *y*, on a log density scale, with the 1, 2 and 3 sigma contours of the cloud
over it.

[`calibration_diagnostics`][pyCamSet.cameras.camera_set.CameraSet.calibration_diagnostics]
is the work every view on this page shares, done once: the reprojection
residuals split off from any lockbox priors, and the detections triangulated
back onto the target. Each plot below is then drawn from it on its own, rather
than all of them together. It is named `fixed` here because the last section
compares it against the free-target solve.

```python exec="true" source="above" session="calibrate"
from pyCamSet.utils.visualisation import cluster_plot, per_camera_coverage

fixed = cams.calibration_diagnostics()
```

<div class="scene-row" markdown>

<div markdown>
```python exec="true" source="above" session="calibrate"
cluster_plot([fixed.residuals])
```
</div>

<div markdown>
```python exec="true" source="above" session="calibrate"
per_camera_coverage(fixed)
```
</div>

</div>

Both axes matter, which is why the left is a 2D histogram rather than a
histogram of the Euclidean error. A calibration that has converged onto the data
leaves a round cloud centred on the origin: the error left over is noise, with
no direction the model could have accounted for and did not. What to look for is
therefore the shape rather than the width — a cloud pulled into an ellipse, or
offset from the origin, or with a second lobe hanging off it, is a systematic
error still in the residual, and no amount of further iteration will remove it.

The right is the same error put back on the sensor it came from, one panel per
camera. It is signed rather than absolute: red is a residual pointing away from
the principal point, blue one pointing back towards it. That sign is what makes
a distortion the lens model has not absorbed visible — it leaves a sensor red at
one radius and blue at another, a ring structure that colouring by unsigned
error hides completely. Noise, by contrast, mixes the two everywhere.

The same distribution as five numbers, without going near the residuals, is on
[`CalibrationReport`][pyCamSet.utils.calibration_report.CalibrationReport]:

```python exec="true" source="above" result="text" session="calibrate"
report = cams.calibration_report

print(f"{report.n_cameras} cameras, {report.n_control_points} control points")
print(f"mean {report.mean_px:.3f} px, p95 {report.p95_px:.3f} px")
print("flags:", report.flags or "none")
```

## Seeing where the cameras ended up

A calibrated set draws itself. This is the quickest check that a calibration is
sane: cameras in roughly the places they physically are, pointing roughly where
they physically point.

<div class="scene-half" markdown>

```python exec="true" source="above" session="calibrate"
cams.plot()
```

</div>

## Seeing where the error is

The two plots above are about the cameras. The two below are about the target,
and they are the pair worth comparing between the fixed-target solve and stage
5's free one.

The scene is every image's view of every feature, carried back into the target's
own frame and piled on top of each other: the size of each cluster is how
consistently that feature was measured, and the shape of the whole cloud is the
target as the cameras believe it to be.

The chart is that scene as two numbers per feature —
[`accuracy_precision_plot`][pyCamSet.utils.visualisation.accuracy_precision_plot].
Accuracy, on *x*, is how far the feature landed from where the target says it
is. Precision, on *y*, is how far its own reconstructions scattered about
wherever they landed. The red diagonal is where the two are equal: a point below
it was measured more tightly than it was placed, which is a target printed wrong
rather than one observed badly.

Left is the calibration that treated the target as known; right is the same
solve with the target's geometry free.

```python exec="true" source="above" log="no" session="calibrate"
from pyCamSet.utils.visualisation import (
    accuracy_precision_plot, target_space_scene)

free = free_cams.calibration_diagnostics()
```

<div class="scene-row" markdown>

<div markdown>
```python exec="true" source="above" session="calibrate"
target_space_scene(fixed, "Fixed target").show()
accuracy_precision_plot(fixed, "Fixed target")
```
</div>

<div markdown>
```python exec="true" source="above" session="calibrate"
target_space_scene(free, "Free target").show()
accuracy_precision_plot(free, "Free target")
```
</div>

</div>

**Read the axes before the shapes.** The two charts are drawn to their own
ranges, and those ranges differ by about a factor of five, so the clouds look
more alike than they are:

```python exec="true" source="above" result="text" session="calibrate"
for name, d in (("fixed", fixed), ("free", free)):
    print(f"{name:>5} target: accuracy {d.accuracy.mean():.3f} mm, "
          f"precision {d.precision.mean():.3f} mm")
```

The fixed-target solve sits close to the diagonal: each feature scatters about
as much as it is displaced, which is what a calibration absorbing fabrication
error into the cameras looks like. Freeing the geometry drops both numbers and
pulls the cloud well below the diagonal — the features are now reproduced far
more tightly than they sit from where they were drawn, which is the signature of
a target that was genuinely printed and folded out of shape.

### Where the target moved

The handler that solved the free target draws how the recovered points moved via
[`special_plots`][pyCamSet.optimisation.standard_bundle_handler.SelfBundleHandler.special_plots],
a diagnosis plot for self optimising cameras.

```python exec="true" source="above" session="calibrate"
free_cams.calibration_handler.special_plots(free_cams.calibration_params)
```

The lattice is formed from the original locations of each feature. Each arrow runs
from a feature's drawn position to where the free solve put it, magnified five
times for visibility.
Unseen features can't be optimised, so aren't drawn.

pyCamSet fixes seven points when optimising the target to remove "gauge freedoms", essentially 
scaling, rotation, and translation of the whole system expressed through bulk motion of the target points.
For use and visualisation, the cameras and target are mapped back to the closest scale and rotation of the target.

This leaves the arrows to represent pure shape variation.
A face bowing away from its plane, or a fold that did not come to a right angle, moves a whole face's worth of features the
same way. This is pretty common for a folded net target!
In a standard calibration, this fabrication error gets blamed on the cameras.

Some arrows disagree with their neighbours: these can be the solve pushing noise
into the geometry instead.
Here, it's likely a few points with limited visibility, but if this plot overall looks worse, this is a case for keeping the fixed-target result.

## Saving

Run through `calibrate_cameras` rather than stage by stage, and — with `save=True`,
the default — the detections, the initial per-camera cameras and the final
optimised set are all cached into the image folder, so a re-run picks up where it
can rather than re-detecting everything. The final set lands at
`optimised_cameras.camset`:

```python
cams = calibrate_cameras("my_rig", target)            # writes into my_rig/
cams = calibrate_cameras("my_rig", target, save_loc="results/")
```

A set can also be saved explicitly at any point:

```python
cams.save("my_cameras.camset")
```

and read back with
[`load_CameraSet`][pyCamSet.utils.saving.load_CameraSet].

## The arguments worth knowing

Every stage above is reachable from the one call, through these:

| Argument | |
|---|---|
| `draw` | Draw each detection as it is made. The fastest way to see that the target being detected is the target you printed |
| `n_lim` | Cap the number of images used for detection. Useful for a quick first pass over a large session |
| `threads` | Detection threads. Defaults to two below your core count, capped at 20 |
| `fixed_params` | Parameters the optimisation is not allowed to move |
| `initial_cams` | Start from a known set rather than solving the per-camera intrinsics again — skips stage 3 |
| `high_distortion` | An iterative scheme for cameras distorted enough that the first detection pass misses features — repeats stages 1 and 3 with the first result in hand |
| `optimise_target` | Run stage 5 |
| `min_detections_per_board` | How many corners a board observation needs before it counts towards stage 3 |
| `problem_options` | Passed through to the solver — `verbosity`, `max_nfev`, outlier rejection, and the gauge |

`verbosity` follows scipy's `least_squares` convention: 0 is silent, 1 reports
termination, 2 reports every iteration and is the default. 3 is a pyCamSet
extension that also shows debug records.

