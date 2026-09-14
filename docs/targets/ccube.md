# Ccube

The Ccube — ChArUco cube — is an easily manufacturable 3D target, designed for
inward-facing camera systems. Functionally it behaves as six rigidly connected
ChArUco targets.

When a calibration target is laminated to a flat surface, manufacturing is
fairly forgiving. Applying separate faces to a 3D object is not: the order and
the orientation of the faces are both easy to get wrong. The Ccube addresses
this by generating the pattern as a **foldable net**, so that the order and
rough orientation of the faces are constrained by the printing itself.

---

## Making one

The length of a Ccube is in millimetres.

```python exec="true" source="above" session="ccube"
from pyCamSet import Ccube

target = Ccube(length=40, n_points=10)
target.plot()
```

Plotting it is how you check that the virtual target and the real one line up
before spending a session photographing the wrong thing.

| Argument | Default | |
|---|---|---|
| `length` | 20.0 | Printed edge of the cube, in millimetres |
| `n_points` | 5 | Corners along one edge of one face |
| `aruco_dict` | `DICT_4X4_1000` | The ArUco dictionary the markers come from |
| `border_fraction` | 0.1 | Blank margin around each face, as a fraction |
| `line_fraction` | 0.003 | Width of the fold lines on the net |
| `legacy` | `False` | OpenCV's pre-4.6 marker origin |
| `marker_backend` | `"aruco1"` | Which detector reads the markers back |

## Calibrating with it

Calibrating with a Ccube is functionally the same as with any other target:

```python
from pathlib import Path

from pyCamSet import Ccube, calibrate_cameras

cams = calibrate_cameras(
    f_loc=Path("my/calibration/path"),
    calibration_target=Ccube(length=40, n_points=10),
    draw=True,
)
```

This is the target used throughout
[Calibrating a camera set](../how-to/calibrate.md), which runs a real one end
to end.

## Printing one

```python
target.save_to_pdf("ccube.pdf")
```

This writes the net at its true dimensions. Print at 100% scale, cut it out,
and fold it over a cube of the `length` it was drawn for.
