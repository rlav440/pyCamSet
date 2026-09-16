# ChArUco2

ChArUco2 is a planar board built on aruco2's own `GridBoard` design: every
square carries an ArUco marker — a standard marker on a black square, an
inverted one on a white square — rather than the sparser, every-other-square
placement `ChArUco` uses. An N x M board of markers has (N+1) x (M+1)
observable intersection corners, including the board's own border.

It has no ArUco 1 (OpenCV) equivalent: OpenCV's `cv2.aruco` module has no
every-square board design under any name, so `ChArUco2` is read with aruco2
only — there is no `marker_backend` to choose.

The design is described in
[this paper](https://www.sciencedirect.com/science/article/pii/S2352711026003249).

!!! note "ChArUco2 is an optional dependency"

    Detection is read with the `aruco2` package, which is not published on
    PyPI. Constructing the target without it raises an `ImportError` naming
    the fix:

    ```bash
    pip install aruco2
    ```

!!! warning "Not yet validated on a real board"

    Every check behind `ChArUco2` runs against aruco2's own rendered raster —
    the corner-id mapping, occlusion, rotation and perspective-warp recovery,
    and the SVG round trip. None of it has been checked against a camera
    photograph of a printed and laminated board. Treat detection quality on a
    real capture as unverified until it has been.

---

## Making one

```python
from pyCamSet import ChArUco2

target = ChArUco2(num_squares_x=10, num_squares_y=10, square_size=4)
target.plot()
```

The arguments that decide where the corners are:

| Argument | Default | |
|---|---|---|
| `num_squares_x`, `num_squares_y` | 5, 5 | Marker squares across and down |
| `square_size` | 10.0 | Printed edge of one square, in millimetres |
| `a_dict` | `DICT_4X4_1000` | The ArUco dictionary the markers come from |

Detection has nothing to tune: `aruco2.detect_grid_board` takes an image, a
board size and a dictionary and nothing else, so there is no
`detection_options` worth setting.

## Calibrating with it

```python
from pathlib import Path

from pyCamSet import ChArUco2, calibrate_cameras

cams = calibrate_cameras(
    f_loc=Path("my/calibration/path"),
    calibration_target=ChArUco2(num_squares_x=10, num_squares_y=10, square_size=4),
    draw=True,
)
```

`draw=True` shows each detection as it is made, which is the quickest way to
see that the board being detected is the board you printed.

## Printing one

```python
target.save_printable("charuco2.svg")
target.save_printable("charuco2.pdf", kind="pdf_vector")
```

Every export kind embeds the exact raster aruco2 both prints and detects
from, at true real-world millimetre scale — there is no separate rendering
path to fall out of sync with what the detector reads. Print at 100% scale —
see the warning under [Choosing a target](index.md#printing).
