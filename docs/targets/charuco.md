# ChArUco

A ChArUco board is a chessboard whose white squares carry ArUco markers. It is
the most robust target here and the simplest to make: print it, laminate it to
something flat, and it is done.

pyCamSet's `ChArUco` wraps OpenCV's own ChArUco board so that it can be used to
jointly calibrate *n* cameras — which the OpenCV framework alone cannot do
beyond three. Partial views still produce valid detections, which is what makes
it work well across a set of cameras that each see a different part of it.

It is also the simplest example of
[implementing a target](../extending/targets.md).

---

## Making one

```python exec="true" source="above" session="charuco"
from pyCamSet import ChArUco

target = ChArUco(num_squares_x=10, num_squares_y=10, square_size=4)
target.plot()
```

The arguments that decide where the corners are:

| Argument | Default | |
|---|---|---|
| `num_squares_x`, `num_squares_y` | 5, 5 | Squares across and down |
| `square_size` | 10.0 | Printed edge of one square, in millimetres |
| `marker_fraction` | 0.8 | How much of a white square the marker fills |
| `a_dict` | `DICT_4X4_1000` | The ArUco dictionary the markers come from |
| `legacy` | `False` | OpenCV's pre-4.6 marker origin, for reading older boards |
| `marker_backend` | `"aruco1"` | Which detector reads the markers back: `"aruco1"` or `"aruco2"` |

`legacy` must match how the physical board was printed: `False` is OpenCV's
current pattern (the default), `True` its pre-4.6 one. pyCamSet does not
switch patterns automatically — both detectors read only the pattern the
board is configured with. If a board's images look like the other pattern,
it logs a once-per-target warning naming the likely correct setting; this
only matters for a board with an even number of rows (`num_squares_y`), since
an odd one reads identically either way.

`marker_backend` describes how the board is read, not what is printed: the
dictionaries offered print identically under both detectors, so one printed
board can be read with either. The GUI, where this target is labelled
**ChArUco1**, therefore asks for no detector in **Create Target…**; it is chosen
in Phase 1 (and in the Optimisation tab), and Phases 2 and 3 use the detector of
the Phase 1 run they continue. See [The graphical workflow](../how-to/gui.md).

Both detectors read the same printed board, so choosing between them is purely
about detection quality, not what to print. In synthetic tests ArUco 1
(OpenCV, the default) gave the lowest corner outlier counts; ArUco 2 found more
corners under blur, noise and strong tilt, with comparable camera pose
accuracy and lens-model fit (fx max 0.0306% aruco2 vs 0.0231% aruco1 —
essentially unchanged for this board).
These are synthetic-test results, not a guarantee for a particular camera or
lighting — if in doubt, try both on your own images.

## Calibrating with it

```python
from pathlib import Path

from pyCamSet import ChArUco, calibrate_cameras

cams = calibrate_cameras(
    f_loc=Path("my/calibration/path"),
    calibration_target=ChArUco(num_squares_x=10, num_squares_y=10, square_size=4),
    draw=True,
)
```

`draw=True` shows each detection as it is made, which is the quickest way to
see that the board being detected is the board you printed.

## Printing one

```python
target.save_printable("charuco.pdf", kind="pdf_vector")
```

Print at 100% scale — see the warning under [Choosing a target](index.md#printing).
