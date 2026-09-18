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
    PyPI: build it from the `third_party/aruco2` submodule, as described
    under "Installing the aruco2 backend" in pyCamSet's `CITATION.md`.
    Constructing the target without it raises an `ImportError` naming the
    fix.

!!! warning "Not yet validated on a real board"

    Every check behind `ChArUco2` runs against rendered images — the
    corner-id mapping, occlusion, rotation and perspective-warp recovery, and
    the SVG round trip. None of it has been checked against a camera
    photograph of a printed and laminated board. Treat detection quality on a
    real capture as unverified until it has been.

---

## Making one

```python exec="true" source="above" session="charuco2"
from pyCamSet import ChArUco2

target = ChArUco2(num_squares_x=10, num_squares_y=10, square_size=4)
target.plot()
```

Every square carries a marker, so a patch of board a good deal smaller than
`ChArUco`'s needs is still enough to say which corners were seen.

The arguments that decide where the corners are:

| Argument | Default | |
|---|---|---|
| `num_squares_x`, `num_squares_y` | 5, 5 | Marker squares across and down |
| `square_size` | 10.0 | Printed edge of one square, in millimetres |
| `a_dict` | `DICT_4X4_1000` | The ArUco dictionary the markers come from |

Detection has nothing to tune: `aruco2.detect_grid_board` takes an image, a
board size, a dictionary and, optionally, the marker ids, and nothing else, so
there is no `detection_options` worth setting.

A board cut by the image edge is still read: pyCamSet pads the image with a
white border before handing it to aruco2 (which otherwise raises on such a
board) and takes the border off the corners it returns. A corner within 10
pixels of the edge is left out, because its sub-pixel refinement reached past
the edge and pulled it off its true position.

aruco2's own corners are then refined and validated before they are returned
(on by default; pass `refine=False`/`validate=False` to
`detect_grid_board_corners` directly to skip either):

- **Refinement** moves each corner to the saddle point of the image, Gaussian
  smoothed at sigma 1.8, with a coarse-to-fine retry for a corner the fine
  scale refuses. Unlike aruco2's own `cornerSubPix`-based refinement, a
  saddle point does not drift with the camera's gamma, and it recovers most
  of the accuracy a blurred or small-squared board otherwise loses (RMS
  1.40 px to 0.21 px on a blurred, low-contrast render, in stage A task A3's
  validation). The four outer corners of the board are not saddle points, so
  they are typically left at aruco2's own position, not a defect. A refined
  corner near an occluder or another non-corner feature is rejected by a
  point-symmetry check and also keeps aruco2's position.
- **Validation** checks every corner against its lattice neighbours (a
  leave-one-out local homography, Tukey-reweighted so a cluster of wrong
  neighbours cannot drag the fit) and drops it if the neighbours disagree, or
  if it has too few of them to check. This is what catches aruco2's own
  occasional false marker detections (see the ghost-marker warning below),
  which otherwise silently place a corner tens of pixels off.

**Pixel convention:** a returned corner is 0.5 px, on both axes, from
aruco2's own raw position -- the same pixel-*corner* convention pyCamSet's
OpenCV-backed `ChArUco`/`Ccube` targets use, so a calibration does not depend
on which target read the image. See `detect_grid_board_corners`'s docstring
for how this is checked.

Two boards are handled at construction. One whose markers include a marker
that reads the same after a half turn (with `DICT_ARUCO_ORIGINAL`, marker 1023,
so a board of 1024 squares) is refused: aruco2 cannot tell which way round that
marker is, and loses the board. A `DICT_4X4_1000` board of 689 squares or more
builds, with a warning: aruco2 can read part of square 688 as marker 17, which
at some image scales loses the whole board. A smaller board, or a larger
dictionary such as `DICT_5X5_1000`, avoids it.

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

The SVG and the vector PDF are true vector, at real-world millimetre scale:
every black marker cell, band tab and corner square is a rectangle in one
filled `<path>`, with no embedded image. Drawn as one shape, abutting cells
meet exactly and rasterise without hairline seams between them. The raster
PDF and `plot()` rasterise the same geometry at the requested `dpi`, so
there is one description of the board behind every output.

That description reproduces aruco2's own board image rather than trusting
a re-implementation of it. Each square holds one whole marker — a one-cell
border around the payload bits, inverted on every other square — and a band
a quarter of a square deep surrounds the board, with black tabs opposite
each white edge square and a black square at each outer corner. The tests
rasterise this layout and compare it pixel for pixel with
`aruco2.get_grid_board_image`, for 4x4, 5x5, 6x6 and 7x7 dictionaries, odd
and even board sizes, and default as well as custom marker ids. The
comparison uses scales at which aruco2's raster puts every edge on a whole
pixel (for a 4x4 dictionary, a bit size divisible by 3); at other scales
aruco2's integer rounding shifts edges by up to a pixel, which is aruco2's
approximation, not the board's geometry. The SVG is also rasterised with
cairo, compared pixel for pixel with that raster at such a scale, and
detected back to every corner at the position its millimetre scale implies.

Print at 100% scale — see the warning under
[Choosing a target](index.md#printing).
