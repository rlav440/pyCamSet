# PuzzleBoardCube

A PuzzleBoardCube puts [PuzzleBoard](puzzleboard.md)'s dense periodic code onto
[Ccube](ccube.md)'s geometry: a cube whose six faces are six **disjoint
windows** of one periodic code. It is a modified implementation of the original
PuzzleBoard, which is a single planar board; the cube version is intended to
work the way a Ccube does, for the same inward-facing rigs.

Because the windows are disjoint, a decoded patch identifies not only which
corners were seen but which face they were on — so a view spanning two faces
resolves correctly rather than folding one face's keys onto another's.

!!! note "PuzzleBoardCube is an optional dependency"

    Like `PuzzleBoard`, it is read by the upstream PuzzleBoard detector:

    ```bash
    pip install "pyCamSet[puzzle]"
    ```

---

## Making one

```python exec="true" source="above" session="puzzlecube"
from pyCamSet import PuzzleBoardCube

target = PuzzleBoardCube(n_points=10, length=200.0)
target.plot()
```

Plotting it is how you check that the virtual target and the real one line up
before spending a session photographing the wrong thing — here that the six
faces carry the windows the printed net carries, in the same order.

| Argument | Default | |
|---|---|---|
| `n_points` | 20 | Corners along one edge of one face. Suggested 10–30 |
| `length` | 200.0 | Printed edge of the cube, in millimetres. Suggested 100–300 |

The `puzzle-cube-v1` layout is **not random**: identical `n_points` and
`length` produce identical face patterns and identical cube geometry, and the
six windows are assigned to the faces in the fixed order front, right, back,
left, top, bottom.

!!! warning "167 squares per side is the ceiling"

    The maximum face size is **167 squares per side**, set by packing six faces
    into the 501-position code field three across and two down. A larger value
    is rejected rather than allowed to overlap or exceed the code period:

    ```text
    ValueError: n_points must not exceed 167 for puzzle-cube-v1
    ```

## Face assignment options

Beyond the detector's own `min_width`, the cube adds two optional stages that
run over the decoded points and decide which face each one belongs to. **Both
are off by default**, pending validation against the full component survey.

```python
target = PuzzleBoardCube(
    n_points=20,
    length=200.0,
    detection_options={
        "plane_consistency_gate": True,
        "face_reassignment": True,
    },
)
```

**`plane_consistency_gate`** runs a two-stage RANSAC homography check per face
and drops points that are not consistent with that face's majority plane. What
it catches is two physically separate regions of the cube decoded into the same
face window — geometrically impossible, and invisible to the solver, which sees
only a set of keys in the wrong place and fits them.

**`face_reassignment`** (which also needs the gate on) does not discard the
dropped cluster outright. A joint two-cluster PnP identifies which other
co-visible face the cluster most likely belongs to, and if the confidence
exceeds `face_reassignment_confidence` the points are relabelled and kept.
Where the signal is not confident the points are dropped, as the gate alone
would have done — the assignment is never forced.

The thresholds underneath these — `plane_gate_contam_squares`,
`plane_gate_min_contam_frac`, `plane_gate_inlier_squares` and the rest — are in
units of square pitch, and their defaults come from the round-2 synthetic
validation rather than from taste. Each carries its own reasoning, readable
without leaving Python:

```python
for parameter in PuzzleBoardCube.detector_parameterisation().parameters:
    print(parameter.key, "--", parameter.concept)
```

!!! warning "The reassignment intrinsics are an assumption, deliberately"

    Face reassignment needs intrinsics to run a PnP, and uses a *generic*
    estimate: a focal length from a typical sensor-size and field-of-view
    assumption, principal point at the image centre. This is deliberate, and
    follows the standing constraint that calibration output must never become a
    functional input to detection. Override them through the
    `face_reassignment_intrinsics_*` parameters if a different generic estimate
    suits your cameras better — but never load them from a `.camset` or any
    other calibration output.

    The PnP confidence gate is also known to invert at low effective
    field-of-view, so treat it with suspicion on narrow-FOV cameras.

## Calibrating with it

```python
from pathlib import Path

from pyCamSet import PuzzleBoardCube, calibrate_cameras

cams = calibrate_cameras(
    f_loc=Path("my/calibration/path"),
    calibration_target=PuzzleBoardCube(n_points=20, length=200.0),
)
```

## Printing one

```python
target.save_printable("puzzle_cube.pdf", kind="pdf_vector")
```

As with the Ccube this writes a foldable net at true scale, so that the order
and rough orientation of the faces are constrained by the printing rather than
by whoever assembles it. Print at 100% scale.

---

## Attribution

The PuzzleBoard source is released under CC0 upstream. Retain the upstream
attribution, and cite the original PuzzleBoard work when publishing results
that use this target.
