# Choosing a target

A calibration target is the digital twin of a real, printed object. It always
provides a way of detecting its own features in an image, and 3D descriptors of
where those features sit in object coordinates. Optionally it also knows how to
draw itself in a 3D plot, and how to produce a printable surface at controlled
dimensions — for a 3D target, that surface is a *net* to be folded over some
underlying geometry.

pyCamSet ships five:

| Target | Shape | Needs | |
|---|---|---|---|
| [`ChArUco`](charuco.md) | planar board | — | The simplest, and the most robust |
| [`ChArUco2`](charuco2.md) | planar board | `aruco2` | Every square marked, aruco2's `GridBoard` design |
| [`Ccube`](ccube.md) | cube, six ChArUco faces | — | For inward-facing rigs |
| [`PuzzleBoard`](puzzleboard.md) | planar board | `puzzle_board` | Dense corners, decoded from a periodic code |
| [`PuzzleBoardCube`](puzzleboard-cube.md) | cube, six PuzzleBoard faces | `puzzle_board` | PuzzleBoard density on Ccube geometry |

Each of them draws itself, which is the quickest way to tell them apart. The
cubes below turn.

<div class="scene-row" markdown>

<div markdown>
```python exec="true" session="targets"
import matplotlib.pyplot as plt

from pyCamSet import ChArUco

plt.figure(figsize=(7, 7))
ChArUco(num_squares_x=10, num_squares_y=10, square_size=4).plot()
```

**[`ChArUco`](charuco.md)** — a chessboard whose white squares carry ArUco
markers, so a partial view still says which corner is which.
</div>

<div markdown>
```python exec="true" session="targets"
from pyCamSet import Ccube

scene = Ccube(n_points=10, length=40).plot(return_scene=True)
scene.window_size = (700, 700)
scene.show()
```

**[`Ccube`](ccube.md)** — six ChArUco faces on a cube, printed as one foldable
net so the faces cannot be assembled in the wrong order.
</div>

</div>

<div class="scene-row" markdown>

<div markdown>
```python exec="true" session="targets"
import matplotlib.pyplot as plt

from pyCamSet import PuzzleBoard

plt.figure(figsize=(7, 7))

target = PuzzleBoard(
    num_squares_x=10,
    num_squares_y=14,
    square_size=2.0,
)
target.plot()
```

**[`PuzzleBoard`](puzzleboard.md)** — a planar board whose every corner is
decoded from the periodic code the pattern is a window onto. Drawn coarse here;
the default is 105x148 corners at a 2 mm pitch, on A4.
</div>

<div markdown>
```python exec="true" session="targets"
from pyCamSet import PuzzleBoardCube

scene = PuzzleBoardCube(n_points=10, length=200.0).plot(return_scene=True)
scene.window_size = (700, 700)
scene.show()
```

**[`PuzzleBoardCube`](puzzleboard-cube.md)** — six disjoint windows of that same
code, one per cube face, on Ccube geometry.
</div>

</div>

The names are also what a target is addressed by in a settings dictionary or on
a GUI form:

```python exec="true" source="above" result="text" session="targets"
from pyCamSet.calibration_targets import TARGET_NAMES

print(TARGET_NAMES)
```

A class is fetched on first use, through
[`target_class`][pyCamSet.calibration_targets.core.target_registry.target_class],
so asking for a ChArUco board does not also import PuzzleBoard's optional
detector.

---

## Calibrating

The main purpose of a target is to calibrate a camera system, which is
[`calibrate_cameras`](../how-to/calibrate.md) and is the same call whatever the
target is:

```python
from pyCamSet import calibrate_cameras, ChArUco

cams = calibrate_cameras(
    f_loc="my/calibration/path",
    calibration_target=ChArUco(num_squares_x=10, num_squares_y=10, square_size=4),
)
```

## Tracking

The other use of a target is as a well characterised, detectable fiducial
marker. With a calibrated camera set, `find_target_pose` returns a bundle
adjustment based estimate of where the target is. The cameras are held at their
calibration, so only the pose of the target is solved for:

```python
from pyCamSet import ChArUco, load_CameraSet

cameras = load_CameraSet("optimised_cameras.camset")
target = ChArUco(num_squares_x=10, num_squares_y=10, square_size=4)

# one instant: a single image from each camera
pose = cameras.find_target_pose({name: images[name] for name in cameras.get_names()}, target)

# a sequence: a list of images per camera, indexed by timestep
poses = cameras.find_target_poses(image_sequences, target)
```

`pose` is a 4x4 homogeneous transform, and `poses` an `(n_timesteps, 4, 4)`
array. Both are also available as
`pyCamSet.optimisation.find_target.find_target_pose_at_timestep` and
`find_target_poses`, if a `CameraSet` method is not what you want.

## Printing

Every target can write itself out at true scale. `save_printable` is the one
call that covers all of them:

```python
target.save_printable("target.svg")               # vector
target.save_printable("target.pdf", kind="pdf_vector")
```

!!! warning "Print at 100% scale"

    The export carries the target's true dimensions. Scaling it to fit the page
    silently invalidates the target: every feature coordinate the calibration
    is told about will be wrong by that scale factor.
