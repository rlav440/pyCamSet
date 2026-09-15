# PuzzleBoard

A PuzzleBoard is a dense planar target. Rather than carrying discrete markers
whose identity is read one at a time, the whole board is one **periodic code**:
any sufficiently large patch of it decodes to its own position in that code, so
a partial, oblique, or heavily cropped view still tells you exactly which
corners were seen.

That is what it buys over a ChArUco board of the same size — many more corners,
and corners that stay identifiable when only a fragment of the board is
visible. What it costs is a dependency, and a board that must be printed at a
fine pitch to be worth it.

!!! note "PuzzleBoard is an optional dependency"

    The detector is the upstream
    [PuzzleBoard repository](https://github.com/PStelldinger/PuzzleBoard) by
    Peer Stelldinger and the HAW Hamburg authors. It is a research codebase and
    is not published on PyPI, so pyCamSet does not pull it in by default:

    ```bash
    pip install "puzzle_board @ git+https://github.com/PStelldinger/PuzzleBoard.git"
    ```

    Constructing the target without it raises an `ImportError` naming that fix.
    See [Troubleshooting](../troubleshooting.md#puzzleboard-is-not-installed).

---

## Making one

The defaults fill an A4 page at a 2 mm pitch. A much smaller board is drawn
here, because at the printed density the pattern reads as grey on a screen:

```python exec="true" source="above" session="puzzleboard"
import matplotlib.pyplot as plt

from pyCamSet import PuzzleBoard

plt.figure(figsize=(5, 7))
target = PuzzleBoard(
    num_squares_x=10,
    num_squares_y=14,
    square_size=2.0,
)
target.plot()
```

Plotting it draws the board that will be printed, zoomed to fill the frame.
Every corner in that pattern is a feature the detector decodes a position from,
which is what the density is for; it is also why a board is worth checking on
screen before it is printed at a pitch this fine.

| Argument | Default | |
|---|---|---|
| `num_squares_x`, `num_squares_y` | 105, 148 | Corners across and down |
| `square_size` | 2.0 | Printed edge of one square, in millimetres |
| `start_x`, `start_y` | 0, 0 | Where in the periodic code this printed window begins |
| `paper_width`, `paper_height` | 210.0, 297.0 | The page it is drawn onto, in millimetres — A4 |

`start_x` and `start_y` are what make two boards distinguishable. The code is
periodic, so a board is a *window* onto it; two boards cut from different
windows decode to different keys and can therefore be told apart, and used
together, without either being mistaken for the other.

## Detecting with it

The detector itself takes one setting, which is not part of what the board *is*
and so is passed separately:

```python
target = PuzzleBoard(detection_options={"min_width": 4})
```

**`min_width`** (default 4) is the smallest decoded grid the detector will
accept. A recovered patch narrower than this is discarded before its position
is decoded. Raising it drops small or oblique views of the board; lowering it
admits patches too small to decode reliably, and risks mislabelled keys.

## Calibrating with it

```python
from pathlib import Path

from pyCamSet import PuzzleBoard, calibrate_cameras

cams = calibrate_cameras(
    f_loc=Path("my/calibration/path"),
    calibration_target=PuzzleBoard(num_squares_x=105, num_squares_y=148, square_size=2.0),
)
```

## Printing one

```python
target.save_printable("puzzleboard.svg")
target.save_printable("puzzleboard.pdf", kind="pdf_vector")
target.save_printable("puzzleboard.pdf", kind="pdf_raster", dpi=600)
```

Prefer a vector format. A dense board at a 2 mm pitch is exactly the case where
a raster export at too low a `dpi` blurs the corners it exists to provide; the
`dpi` argument is ignored by the vector formats and only applies to
`pdf_raster`.

Print at 100% scale — see the warning under
[Choosing a target](index.md#printing).

---

## Attribution

The PuzzleBoard source is released under CC0 upstream. Retain the upstream
attribution, and cite the original PuzzleBoard work when publishing results
that use this target.
