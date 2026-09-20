# CIco2

The CIco2 — ChArUco2 icosahedron — is the [CIco](cico.md) with every face
printed as a [ChArUco2](charuco2.md) board: aruco2's `GridBoard` design, with
an ArUco marker on every square rather than on every other one. The
icosahedron itself is CIco's — the same face transforms, the same foldable net,
the same face numbering. The GUI labels it **ChArUco2 icosahedron**.

!!! note "CIco2 is an optional dependency"

    Its faces are read with the `aruco2` package, which is not published on
    PyPI: build it from the `third_party/aruco2` submodule, as described under
    "Installing the aruco2 backend" in pyCamSet's `CITATION.md`. Constructing
    the target without it raises an `ImportError` naming the fix.

!!! warning "Not yet validated on a real printed target"

    Every check behind `CIco2` runs against rendered images. None of it has
    been checked against a camera photograph of a printed and assembled
    icosahedron. Treat detection quality on a real capture as unverified until
    it has been.

---

## Making one

```python exec="true" source="above" session="cico2"
from pyCamSet import CIco2

target = CIco2(length=100, n_points=12)
target.plot()
```

| Argument | Default | |
|---|---|---|
| `length` | 100.0 | Printed edge of one face, in millimetres, border included |
| `n_points` | 12 | Squares along the bottom edge of a face |
| `aruco_dict` | `DICT_4X4_1000` | The ArUco dictionary the markers come from |
| `border_fraction` | 0.15 | Blank margin around each face's board, as a fraction of the face |

There is no `legacy` and no `marker_backend`: a grid board has no legacy
layout and no ArUco 1 equivalent, so a CIco2 is read with ArUco 2 only.

## Why ChArUco2 and not ChArUco, on a clipped face

A marker on every square is what lets a grid board report the corners on its
**outer boundary** as well as its interior ones — every corner of a printed
square is localisable from that square's own marker. On a full rectangle that
is a modest gain. On a face clipped to a triangle it is most of the target,
because clipping throws away a board's outside, which is exactly where the
difference lies:

| `n_points` | Printed squares | Corners, ChArUco1 rule | Corners, ChArUco2 rule |
|---|---|---|---|
| 7 | 13 | 4 | 24 |
| 8 | 18 | 8 | **30** |
| 10 | 30 | 17 | 45 |

At twelve squares to an edge a CIco2 offers 65 corners a face — 1300 points in
total — against a CIco's 29. The advantage is largest where the boundary is
most of the board.

## The band, on a staircase

aruco2 draws a ring of alternating tabs around a rectangular board. That ring
is what gives the board's outer corners a black side and a white one; without
it they are board on one side and blank paper on the other, and there is no
corner to find.

A clipped board's outer boundary is a **staircase**, not a rectangle, so the
ring cannot be drawn as it stands. The rule it follows — a tab outward from a
square whose `(column + row)` is odd — is applied instead to every edge of a
printed square that meets one that was clipped away, with the same square
filling the outward diagonal where two such edges meet. `CIco2` prints that
staircase band, and every boundary corner is recovered because of it.

## Marker ids, and what clipping would otherwise cost

Every square carries its own marker, and every square of the **rectangle** the
board is cut from needs an id whether or not it is printed — the board that
reads a face *is* that rectangle. Given a block each, twenty faces would spend
more than half the alphabet on squares the triangle cut away: at eight squares
to an edge, 22 of every 40.

So the squares that are never printed do not get a block each. **They share one
block of filler ids between all twenty faces.** That is safe precisely because
they are never printed — no image can contain one, so no two faces can be
confused by one. A single repeated filler would be simpler still, and aruco2
refuses it: a board must carry a distinct id per square. A shared *block*
satisfies that while costing one alphabet rather than twenty.

| `n_points` | Board | Printed / face | Filler (shared) | A block each | Shared filler |
|---|---|---|---|---|---|
| 8 | 8 × 5 | 18 | 22 | 800 | **382** |
| 10 | 10 × 6 | 30 | 30 | 1200 | **630** |
| 12 | 12 × 8 | 46 | 50 | 1920 | **970** |
| 16 | 16 × 12 | 90 | 102 | 3840 | 1902 — refused |

That is what lets the default be twelve squares to an edge inside an ordinary
thousand-marker dictionary, where a block each would need 1920. Past the
ceiling, constructing one raises a `ValueError` saying how many markers were
needed and how many the dictionary holds.

## Printing one

`save_printable` writes the twenty faces as one foldable net; `to_stl` writes
the bare solid to print a core to mount them on, built from the same face
transforms the target's points are.

```python
target.save_printable("cico2_net", kind="pdf_vector")
target.to_stl("cico2_core")
```

### How that is checked

- Rasterising the printed net and reading it back recovers **all twenty faces,
  with every one of the 65 corners on every face** —
  `test_the_printed_net_gives_back_every_face`.
- The boundary corners — more than half of a clipped board's corners — are all
  found, which is what the staircase band is for —
  `test_the_band_keeps_the_corners_on_a_clipped_board_s_edge`.
- Corners land within a fiftieth of a square of where `point_data` says they
  are — `test_a_printed_face_puts_its_corners_where_it_says_they_are`.
- No two faces share a *printed* marker, and the only ids they do share are the
  filler — `test_only_the_squares_a_face_prints_take_an_id_of_their_own`.
- No face prints a filler marker, which is the whole reason sharing them is
  safe — `test_a_shared_filler_marker_is_never_printed`.
