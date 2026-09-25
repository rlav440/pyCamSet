# PuzzleBoardIco

The PuzzleBoardIco is twenty triangular [PuzzleBoard](puzzleboard.md) faces on
an icosahedron — [PuzzleBoardCube](puzzleboard-cube.md)'s arrangement on a
solid with twenty faces instead of six. The GUI labels it **PuzzleBoard
icosahedron**.

As on the cube, each face prints a different window of PuzzleBoard's periodic
code field, so a decoded position says both which face was seen and where on
it. There are no markers to share out and no dictionary to run out of, which
is why this is the densest of the icosahedral targets: at sixteen squares to an
edge a face carries 93 corners, **1860 points in total**.

!!! note "PuzzleBoardIco is an optional dependency"

    It is read with the `puzzle_board` package, which is not published on PyPI:
    `pip install 'puzzle_board @ git+https://github.com/PStelldinger/PuzzleBoard.git'`.
    Constructing the target without it raises an `ImportError` naming the fix.

!!! warning "Not yet validated on a real printed target"

    Every check behind `PuzzleBoardIco` runs against rendered images. None of
    it has been checked against a camera photograph of a printed and assembled
    icosahedron. Treat detection quality on a real capture as unverified until
    it has been.

---

## Making one

```python exec="true" source="above" session="puzzleboard-ico"
from pyCamSet import PuzzleBoardIco

target = PuzzleBoardIco(length=100, n_points=16)
target.plot()
```

| Argument | Default | |
|---|---|---|
| `length` | 100.0 | Printed edge of one face, in millimetres |
| `n_points` | 16 | Code squares along the bottom edge of a face |

## Why the windows are simply tiled

It would be reasonable to expect that spacing the twenty windows apart in the
code field buys margin against a patch of one face decoding as another.
**Measured, it does not.** With twenty 20×20 windows, the smallest Hamming
distance between a 3×3 patch of one face and a 3×3 patch of another is 1 — for
a row-major tiling, for a maximally spread layout, and for random disjoint
layouts alike.

The reason is what the code is. PuzzleBoard's base code is a *sub-perfect map*,
built so that every patch is unique; it says nothing about how far apart they
are, and no choice of window can add a property the code does not have.

What placement **does** decide is whether the windows overlap. Two faces
sharing any of the field carry literally identical patches — a distance of
zero — and a patch in the shared part decodes to both. That is the one
placement failure worth designing against, so the windows are tiled disjointly
and deterministically, and the layout is frozen as `puzzle-ico-v1`.

Robustness is left where PuzzleBoard itself puts it: in voting across the many
patches a face shows at once, not in the margin of any one of them.

## Geometry

The window is clipped to the triangle, squares and all: a square the triangle
cuts through is printed as the part of it that fits. A cut square still meets
its neighbours in corners, and a chessboard corner is the whole of what the
detector wants, so there is nothing to gain by dropping it — and a good deal to
lose, because dropping it costs the corners along the entire edge of the face.
At sixteen squares to an edge the pattern covers the face and carries **93
corners, against the 65 whole squares alone would leave**.

A circle carries a bit inside a square rather than at its corners, so unlike a
square it does not survive being cut: one is printed only where it fits whole.
Where a circle is missing its bit reads as noise and is outvoted, which is
where PuzzleBoard puts its robustness anyway.

### Two things the clip has to respect

**The pattern stops a fifth of a square short of the face's edge.** Two faces
that share an edge in the net sit side by side across the fold, their lattices
at sixty degrees to each other. Printed flush they join into *one* grid under
the detector and the decode goes wrong — rasterised and read back, seven of the
twenty faces came out as nothing and others as more points than they have.
Holding each face's pattern back leaves a white channel two fifths of a square
wide down every fold, and every face then reads back complete.

**A corner is kept only 0.15 of a square inside what was printed.** A lattice
corner is a chessboard corner wherever the pattern surrounds it, but a corner
sitting on the edge of the pattern is half a junction and is not found. Keeping
object points for corners the detector cannot see would leave the target
claiming points that never arrive; 0.15 of a square is where, rendered and read
back, nothing kept is missed — from six squares to an edge up to the largest
window the code field holds.

!!! note "A half-square convention worth knowing about"

    Upstream PuzzleBoard centres its squares on the integer code positions, so
    the corner it reports for code `(row, col)` falls at a half-integer. The
    squares here run integer to integer instead, which is the convention the
    shared clipping works in and which puts the corners on the integers.

    That half-square difference is a whole number of code positions — 334,
    which is 2 × 167, the base code's period — and it is the same number
    whichever window is printed and wherever it is placed. The code is read
    that far back so a printed window decodes to the position it was cut from.
    Get it wrong and every face decodes as a different one, silently, so a test
    pins it.

## Printing one

`save_printable` writes the twenty faces as one foldable net; `to_stl` writes
the bare solid to print a core to mount them on, built from the same face
transforms the target's points are.

```python
target.save_printable("puzzleboard_ico_net", kind="pdf_vector")
target.to_stl("puzzleboard_ico_core")
```

### How that is checked

- No two faces share any of the code field —
  `test_no_two_faces_share_any_of_the_code_field`.
- A printed window decodes to the window it was cut from, which is what the
  half-square phase is for —
  `test_a_printed_window_decodes_to_the_window_it_was_cut_from`.
- Rasterising the printed net and reading it back recovers **all twenty faces,
  with every one of the 93 corners on every face** —
  `test_the_printed_net_gives_back_every_face`. This is also what pins the
  channel down the folds: printed flush, this test is what fails.
- Nothing printed runs past the fold, and every corner the target keeps has
  printed pattern around it —
  `test_nothing_printed_runs_past_where_the_face_is_folded`,
  `test_every_corner_a_face_keeps_has_printed_pattern_around_it`.
