# Calibrating a camera set

[`calibrate_cameras`][pyCamSet.calibration.camera_calibrator.calibrate_cameras]
coordinates the whole job, from detection through to a finished `CameraSet`.
It is one call, and it is the path this library is built around.

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

## The call

Everything below runs against the three-camera Ccube corpus that ships with
this repository, as the documentation is built — so the numbers and the scenes
on this page are what this code actually produced, not a transcript of what it
once produced.

```python exec="true" source="above" session="calibrate"
from cv2 import aruco

from pyCamSet import Ccube, calibrate_cameras

target = Ccube(
    n_points=10,
    length=40,
    aruco_dict=aruco.DICT_6X6_1000,
    border_fraction=0.2,
)

cams = calibrate_cameras(
    "tests/test_data/calibration_ccube",
    target,
    save=False,
    threads=1,
)
```

`save=False` keeps it from writing its results back beside the images, which is
what you want here and almost never what you want in real use — see
[Saving](#saving) below.

!!! note "Why `threads=1` here"

    Detection across several threads uses a process pool, and on any platform
    that spawns rather than forks, each worker re-imports the module it was
    started from. That is fine from a script guarded by
    `if __name__ == "__main__":`, and it is what you should use for real work —
    the default is two below your core count. It is not fine from inside a
    documentation build, whose main module is the build itself, so this page
    detects sequentially.

## Reading the result

The output above is what pyCamSet pushes to your terminal while it works, and
it comes in three reports.

**The detection report** says what was found in the images before anything was
solved. This is the one that catches a session that was never going to work: a
camera that saw the target in no image cannot be calibrated, and a board seen
in only a handful of images constrains almost nothing.

**The intrinsics report** covers the per-camera calibrations, each solved
independently, which are what the bundle adjustment starts from. Wild focal
lengths or distortion coefficients here mean the joint solve is starting from
somewhere it may not recover from.

**The calibration summary** is the bundle adjustment itself, and the block to
read when judging a finished calibration:

- **Reprojection error** — `initial` is the mean Euclidean error before
  optimising and `final mean` after, so the gap between them is what the solve
  achieved. `median`, `rms`, `p95` and `max` describe the distribution: a
  median far below the mean means a small number of bad points are carrying the
  average.
- **The per-camera table** grades each camera's error by colour — under 0.1 px
  is exceptional, under 1 px good, under 5 px poor, above that bad. One camera
  far worse than the rest is a camera to go and look at.
- **worst images** names the individual images contributing most of the error,
  which is where to look for a blurred frame or a target caught mid-move.
- **Flags** are the concerns worth a person's attention, raised by the run
  itself.

The same numbers are available as data rather than as text, on
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

```python exec="true" source="above" session="calibrate"
cams.plot()
```

## Seeing where the error is

`visualise_calibration` is the per point, per camera and per image detail
underneath the summary — the residual distributions, and where on the sensor
the error falls.

```python exec="true" source="above" session="calibrate"
cams.visualise_calibration()
```

## Saving

By default — `save=True` — the calibration caches its detections, the initial
per-camera cameras, and the final optimised set into the image folder, so a
re-run picks up where it can rather than re-detecting everything. The final set
lands at `optimised_cameras.camset`:

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

| Argument | |
|---|---|
| `draw` | Draw each detection as it is made. The fastest way to see that the target being detected is the target you printed |
| `n_lim` | Cap the number of images used for detection. Useful for a quick first pass over a large session |
| `threads` | Detection threads. Defaults to two below your core count, capped at 20 |
| `fixed_params` | Parameters the optimisation is not allowed to move |
| `initial_cams` | Start from a known set rather than solving the per-camera intrinsics again |
| `high_distortion` | An iterative scheme for cameras distorted enough that the first detection pass misses features |
| `min_detections_per_board` | How many corners a board observation needs before it counts towards the initial per-camera calibration |
| `problem_options` | Passed through to the solver — `verbosity`, `max_nfev`, outlier rejection, and the gauge |

`verbosity` follows scipy's `least_squares` convention: 0 is silent, 1 reports
termination, 2 reports every iteration and is the default. 3 is a pyCamSet
extension that also shows debug records.

---

## Next

If the finished calibration is limited by the target rather than by the
cameras — a printed board that is not quite the shape it was drawn as — then
[self-calibration](self-calibration.md) is what to do about it.
