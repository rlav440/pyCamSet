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
| `n_points` | 5 | Squares along one edge of one face |
| `aruco_dict` | `DICT_4X4_1000` | The ArUco dictionary the markers come from |
| `border_fraction` | 0.1 | Blank margin around each face, as a fraction |
| `line_fraction` | 0.003 | Width of the fold lines on the net |
| `legacy` | `False` | OpenCV's pre-4.6 marker origin |
| `marker_backend` | `"aruco1"` | Which detector reads the markers back: `"aruco1"` or `"aruco2"` |

`legacy` must match how the physical cube was printed: `False` is OpenCV's
current pattern (the default), `True` its pre-4.6 one. pyCamSet does not
switch patterns automatically — both detectors read only the pattern each
face is configured with. If a face's images look like the other pattern, it
logs a once-per-target warning naming the face and the likely correct
setting; this only matters for a face with an even number of rows
(`n_points`), since an odd one reads identically either way.

`marker_backend` describes how the cube is read, not what is printed: the
dictionaries offered print identically under both detectors, so one printed
cube can be read with either. The GUI, where this target is labelled
**ChArUco1 ccube**, therefore asks for no detector in **Create Target…**; it is chosen
in Phase 1 (and in the Optimisation tab), and Phases 2 and 3 use the detector of
the Phase 1 run they continue. See [The graphical workflow](../how-to/gui.md).

Both detectors read the same printed cube, so choosing between them is purely
about detection quality, not what to print. In synthetic tests ArUco 1
(OpenCV, the default) gave the lowest corner outlier counts; ArUco 2 found more
corners under blur, noise and strong tilt, with comparable camera pose
accuracy, but produced more corners over 3 px off and a worse lens-model fit on
some cube datasets. These are synthetic-test results, not a guarantee for a
particular camera or lighting — if in doubt, try both on your own images.

For accuracy, use at least 10 squares per face side (`n_points`): 5-square
faces were 3–11x worse in rotation, translation and focal-length error in
synthetic tests, for both Ccube and [Ccube2](ccube2.md).

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
