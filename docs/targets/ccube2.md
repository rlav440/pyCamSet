# Ccube2

The Ccube2 — ChArUco2 cube — is the [Ccube](ccube.md) with every face printed
as a [ChArUco2](charuco2.md) board: aruco2's `GridBoard` design, with an ArUco
marker on every square rather than on every other one. The cube itself is
Ccube's — the same face transforms, the same foldable net, the same order and
orientation of faces — so a Ccube2 folds and mounts exactly as a Ccube does.
The GUI labels it **ChArUco2 ccube**.

Functionally it behaves as six rigidly connected ChArUco2 boards. Each face is
a board of its own, with its own marker ids, so each is found independently of
the others.

!!! note "Ccube2 is an optional dependency"

    Its faces are read with the `aruco2` package, which is not published on
    PyPI: build it from the `third_party/aruco2` submodule, as described under
    "Installing the aruco2 backend" in pyCamSet's `CITATION.md`. Constructing
    the target without it raises an `ImportError` naming the fix.

!!! warning "Not yet validated on a real cube"

    Every check behind `Ccube2` runs against rendered images: each face against
    aruco2's own board image, the SVG net rasterised by cairo, and a synthetic
    pinhole photograph of the cube. None of it has been checked against a camera
    photograph of a printed and folded cube. Treat detection quality on a real
    capture as unverified until it has been.

---

## Making one

The length of a Ccube2 is in millimetres.

```python
from pyCamSet import Ccube2

target = Ccube2(length=40, n_points=6)
target.plot()
```

| Argument | Default | |
|---|---|---|
| `length` | 20.0 | Printed edge of the cube, in millimetres, border included |
| `n_points` | 5 | Marker squares along one edge of one face |
| `aruco_dict` | `DICT_4X4_1000` | The ArUco dictionary the markers come from |
| `border_fraction` | 0.1 | Blank margin around each face's board, as a fraction of the face |
| `line_fraction` | 0.003 | Width of the fold lines on the net |

There is no `legacy` and no `marker_backend`. A grid board has no legacy
layout, and it has no ArUco 1 (OpenCV) equivalent, so a Ccube2 is read with
ArUco 2 only. In the GUI's Phase 1 choosing a ChArUco2 ccube selects
**ArUco 2 (aruco2)** by itself and greys out ArUco 1; see
[The graphical workflow](../how-to/gui.md).

For accuracy, use at least 10 squares per face side (`n_points`): 5-square
faces were 3–11x worse in rotation, translation and focal-length error in
synthetic tests, for both [Ccube](ccube.md) and Ccube2.

## Geometry

`n_points` means what it means for a Ccube: **squares per face side**. Every
square carries one marker, so a face holds `n_points²` markers and has
`(n_points + 1)²` corners, the board's own outer border included.

A face of edge `length` is part margin and part board:

- the square size is `length × (1 − border_fraction) / n_points`;
- the board starts `length × border_fraction / 2` in from each edge of the face.

Around its board a grid board prints a band a quarter of a square deep, with
black tabs opposite the white edge squares and a black square at each outer
corner. On a Ccube2 that band is drawn in the face's margin, so the margin must
be at least as deep as the band. That is the case when

```
border_fraction ≥ 1 / (2 × n_points + 1)
```

— for example 1/9 ≈ 0.111 for a 4-square face, 1/11 ≈ 0.091 for the default 5,
and 1/17 ≈ 0.059 for 8. A smaller border would push the tabs round the fold onto
the neighbouring face, and the constructor refuses it with the minimum for that
`n_points` in the message.

A corner's `point_data` key is `[face, gid]`, with `gid = row × (n_points + 1) +
col` counted row-major from the face's top-left corner — the same corner id
aruco2 reports for a board.

## Marker ids and dictionary capacity

Face `k` (0 to 5) uses the ids `k × n² … (k + 1) × n² − 1`, row-major, where
`n = n_points`. The ranges do not overlap, so no marker on one face can be taken
for a marker on another. aruco2's grid-board detector finds one board per call,
and only under the ids it was printed with, so detection runs it once per face,
six times per image.

The whole cube needs `6 × n²` distinct markers from one dictionary, and the
constructor refuses a cube its dictionary cannot fill. The largest face each
dictionary size allows:

| Dictionary size | Examples | Largest `n_points` |
|---|---|---|
| 50 | `DICT_4X4_50`, `DICT_5X5_50` | 2 |
| 100 | `DICT_4X4_100`, `DICT_6X6_100` | 4 |
| 250 | `DICT_4X4_250`, `DICT_ARUCO_MIP_36h12` | 6 |
| 256 | `DICT_ALVAR_5X5_256` | 6 |
| 1000 | `DICT_4X4_1000` (the default), `DICT_ALVAR_7X7_1000` | 12 |
| 1024 | `DICT_ARUCO_ORIGINAL` | 13 |

The dictionaries offered are ChArUco2's: AprilTag dictionaries are left out.

## Calibrating with it

Calibrating with a Ccube2 is the same call as with any other target:

```python
from pathlib import Path

from pyCamSet import Ccube2, calibrate_cameras

cams = calibrate_cameras(
    f_loc=Path("my/calibration/path"),
    calibration_target=Ccube2(length=40, n_points=6),
    draw=True,
)
```

As for ChArUco2, detection has nothing to tune: `aruco2.detect_grid_board` takes
an image, a board size, a dictionary and the ids, and nothing else. A face cut
by the image edge is read as a ChArUco2 board is: its corners more than 10
pixels inside the image are returned, and one face aruco2 fails on is left out
of that image with a warning rather than stopping the calibration.

Each face is refined and validated exactly as a ChArUco2 board is (see
[ChArUco2's "Calibrating with it"](charuco2.md#calibrating-with-it) for what
that does and why), since both go through the same
`detect_grid_board_corners` choke point, once per face. The returned corners
are in the same pixel-*corner* convention as every other ArUco-based target
in pyCamSet, 0.5 px from aruco2's own raw position.

## Printing one

```python
target.save_printable("ccube2.svg")
target.save_printable("ccube2.pdf", kind="pdf_vector")
```

The SVG and the vector PDF are true vector, at real-world millimetre scale. Each
face is drawn from the same geometry a ChArUco2 board prints from, as one filled
`<path>` of rectangles placed in Ccube's net, with no embedded image, so abutting
cells rasterise without seams. The raster PDF and `plot()` use face textures
rasterised from that same geometry. With `individual_faces=True` the PDF has one
face per page instead of the net, and those pages are raster from the face
textures even when `kind="pdf_vector"` is asked for; a warning says so.

The face numbers that tell you how to fold the net are printed in the white part
of the band below each face's board, under a square with no tab beneath it, half
the band tall and centred in it, so a number touches no marker, tab or corner
square.

How that is checked:

- each face texture, without its number, is compared pixel for pixel with
  `aruco2.get_grid_board_image` for that face's ids, for 4x4, 5x5 and 6x6
  dictionaries, at scales where aruco2's raster puts every edge on a whole pixel;
- every face texture is detected under its own ids with all of its corners, at
  the positions its geometry implies, including at the smallest border the band
  fits in;
- the SVG net is rasterised by cairo, with the cut outline and face numbers drawn,
  and all six faces are detected with every corner where the net places them; the
  numbers are checked to fall only on white, inside their band;
- a synthetic pinhole camera photographs the cube from three directions, three
  faces at a time, and every detected corner lands within a pixel of where its
  `point_data` projects.

Print at 100% scale — see the warning under
[Choosing a target](index.md#printing) — cut the net out, and fold it over a cube
of the `length` it was drawn for.
