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
| `marker_backend` | `"aruco1"` | Which detector reads the markers back |

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
