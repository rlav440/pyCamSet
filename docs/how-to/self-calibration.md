# Self-calibration

A normal calibration treats the target as known. Every feature sits exactly
where the target says it does, and only the cameras and the target's poses are
solved for. That is the right assumption when the target is more accurate than
the cameras — and the wrong one when it is not.

**Self-calibration frees the target's own points.** The feature coordinates
become parameters of the optimisation, so the solve can absorb the difference
between the target as drawn and the target as printed, laminated, folded and
handled.

On the Ccube corpus in this repository that is not a marginal effect, and the
numbers below are what it actually achieves — this page runs the comparison as
the site is built.

---

## Fixing the gauge

Letting every point move makes the problem under-determined in seven degrees of
freedom: the whole scene can translate, rotate, and scale without changing a
single reprojection.
[`SelfBundleHandler`][pyCamSet.optimisation.standard_bundle_handler.SelfBundleHandler]
breaks that symmetry for you, by holding seven parameters of three non-colinear
target points fixed. You do not have to choose them, but it is worth knowing it
happened — the result is gauged against those points, so a self-calibrated
target is only as correctly *scaled* as the points that were held.

## Running one

Self-calibration is a second bundle adjustment over the detections a first
calibration already made, so it starts from a finished `CameraSet`:

```python exec="true" source="above" session="selfcal"
from multiprocessing import cpu_count

from cv2 import aruco

from pyCamSet import Ccube, calibrate_cameras
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

target = Ccube(
    n_points=10,
    length=40,
    aruco_dict=aruco.DICT_6X6_1000,
    border_fraction=0.2,
)
cams = calibrate_cameras(
    "tests/test_data/calibration_ccube", target, save=False, threads=1)

handler = SelfBundleHandler(
    detection=cams.calibration_handler.detection,
    target=target,
    camset=cams,
    options={"max_nfev": 100},
)
handler.set_from_templated_camset(cams)

_, self_calibrated = run_bundle_adjustment(handler, threads=cpu_count())
```

Three things are worth pointing at in that block:

- **`cams.calibration_handler.detection`** — the detections are reused rather
  than remade. Detection is the expensive half of a calibration, and the
  self-calibration is solving from exactly the same observations.
- **`set_from_templated_camset`** — seeds the free point geometry from the
  template the first calibration used, so the solve starts at the target as
  drawn and moves away from it only as the data demands.
- **`options={"max_nfev": 100}`** — a cap on solver evaluations. The problem is
  much larger than the fixed-target one, because every feature coordinate is now
  a parameter.

The output above is two calibration summaries: the fixed-target solve, then the
free-target one. `run_bundle_adjustment` returns the optimisation result and the
finished `CameraSet`.

## Comparing the two

The thing worth knowing is whether letting the shape move helped:

```python exec="true" source="above" result="text" session="selfcal"
before = cams.calibration_report.mean_px
after = self_calibrated.calibration_report.mean_px

print(f"fixed target: {before:7.3f} px")
print(f"free target:  {after:7.3f} px")
print(f"improvement:  {before / after:7.1f}x")
```

If the error barely moves, the target was not the limiting factor and the
fixed-target calibration is the one to keep — it is gauged by a known object
rather than by three of its own points, which is a stronger claim about scale.

If the error drops sharply, as it does here, the target was the limiting factor:
most of what looked like camera error was printing and assembly error.

## Seeing how far the target moved

The self-calibrated geometry can be drawn against the cameras that produced it:

```python exec="true" source="above" session="selfcal"
self_calibrated.plot()
```

A target whose points have moved a long way from where they were drawn is worth
being suspicious of, rather than pleased about: a large motion with a large
error reduction is a target that was genuinely misprinted, but a large motion
with only a small error reduction is usually the solve absorbing something else
— a systematic detection bias, or an image set too small to constrain the extra
parameters.

---

## Doing it without writing the handler yourself

The graphical workflow runs exactly this as its Phase 4, with diagnostics that
report how far the target moved, how flat it stayed, and how the result compares
against the fixed-target solve. See
[The graphical workflow](gui.md#phase-4-self-calibration).
