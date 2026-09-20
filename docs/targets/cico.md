# CIco

The CIco — ChArUco icosahedron — is twenty triangular [ChArUco](charuco.md)
faces on an icosahedron, the same idea as the [Ccube](ccube.md) on a solid with
twenty faces instead of six. The GUI labels it **ChArUco1 icosahedron**.

The reason for twenty faces is what a camera can see at once. A distant camera
sees three faces of a cube; it sees **exactly ten faces of an icosahedron**,
from any direction whatever — which is asserted rather than claimed, over 500
random viewing directions, in `tests/test_polyhedra.py`. More of the target is
visible to more of a rig at a time, and the orientations it presents as it is
moved are much more finely spaced.

Functionally it behaves as twenty rigidly connected ChArUco boards. Each face
is a board of its own, cut from its own slice of the marker dictionary, so each
is found independently of the others.

!!! warning "Not yet validated on a real printed target"

    Every check behind `CIco` runs against rendered images: a face against its
    own printed pattern, and the whole SVG net rasterised by cairo and read
    back. None of it has been checked against a camera photograph of a printed
    and assembled icosahedron. Treat detection quality on a real capture as
    unverified until it has been.

---

## Making one

The length of a CIco is the edge of one triangular face, in millimetres.

```python exec="true" source="above" session="cico"
from pyCamSet import CIco

target = CIco(length=100, n_points=10)
target.plot()
```

| Argument | Default | |
|---|---|---|
| `length` | 100.0 | Printed edge of one face, in millimetres, border included |
| `n_points` | 10 | Chessboard squares along the bottom edge of a face |
| `aruco_dict` | `DICT_4X4_1000` | The ArUco dictionary the markers come from |
| `border_fraction` | 0.1 | Blank margin around each face's board, as a fraction of the face |
| `legacy` | `False` | Which of OpenCV's two marker layouts the faces were printed to |

## Geometry

A square lattice does not fit a triangle. The board is laid over the face and
**clipped to whole squares**: a square wholly inside the triangle is printed, a
square the edge crosses is not. Nothing downstream is told, because a board
with squares missing is a partly hidden board, and reading one of those is what
a ChArUco detector already does.

Clipping keeps far more of a face than the alternative. The largest rectangle
that fits inside a triangle covers half of it whatever its size; clipping to
whole cells keeps 51% of a face at six squares to an edge, rising to 84% at
twenty.

A corner is only kept where **all four** of its squares were printed — a corner
with one of its squares missing is the end of a line, not a corner. At ten
squares to an edge that leaves 30 printed squares and 17 corners per face, so
340 points in total, of which about 170 face any given viewpoint.

## Marker ids and dictionary capacity

The dictionary is split twenty ways, one slice per face, by the same
`split_aruco_dictionary` a [Ccube](ccube.md) uses: face `k`'s markers are the
parent dictionary's `k * markers_per_face` onwards, and each face's board
numbers its own from zero.

Every square of the **rectangle the board is cut from** needs a marker, printed
or not, because the board object that reads a face is that rectangle. A
thousand-marker dictionary therefore runs out at about twelve squares to an
edge:

| `n_points` | Squares reserved per face | Twenty faces | Corners per face |
|---|---|---|---|
| 8 | 20 | 400 | 8 |
| 10 | 30 | 600 | 17 |
| 12 | 48 | 960 | 29 |

Past that, constructing one raises a `ValueError` saying how many markers were
needed and how many the dictionary holds.

## Printing one

`save_printable` writes the twenty faces as one foldable net, as SVG or as a
vector or raster PDF, at true millimetre scale.

```python
target.save_printable("cico_net", kind="pdf_vector")
```

A twenty-faced net folded by hand is only as accurate as the folding, and what
a calibration measures is the solid, not the paper. `to_stl` writes the bare
icosahedron to print a core to mount the faces on:

```python
target.to_stl("cico_core")
```

The written solid is built from the **same face transforms** the target's own
points are, so the printed core cannot disagree with the geometry the
calibration assumes — a mistake that would otherwise only surface after the
target had been printed, assembled and calibrated against.

### How that is checked

- The net lays all twenty faces out without overlapping, and each face in the
  net is the size it is on the solid — `tests/test_polyhedra.py`.
- Rasterising the printed net and reading it back recovers **all twenty faces,
  with every corner on every face** — `test_the_printed_net_gives_back_every_face`.
- A face read on its own decodes as itself and no other, and its corners land
  within a fiftieth of a square of where `point_data` says they are —
  `test_a_printed_face_puts_its_corners_where_it_says_they_are`.
- The STL's facets match the face transforms the target calibrates against —
  `test_the_solid_writes_itself_for_printing`.
