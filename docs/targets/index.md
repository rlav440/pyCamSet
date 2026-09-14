# Choosing a target

A calibration target is the digital twin of a real, printed object. It always
provides a way of detecting its own features in an image, and 3D descriptors of
where those features sit in object coordinates. Optionally it also knows how to
draw itself in a 3D plot, and how to produce a printable surface at controlled
dimensions — for a 3D target, that surface is a *net* to be folded over some
underlying geometry.

pyCamSet ships four:

| Target | Shape | Needs | |
|---|---|---|---|
| [`ChArUco`](charuco.md) | planar board | — | The simplest, and the most robust |
| [`Ccube`](ccube.md) | cube, six ChArUco faces | — | For inward-facing rigs |
| [`PuzzleBoard`](puzzleboard.md) | planar board | `pyCamSet[puzzle]` | Dense corners, decoded from a periodic code |
| [`PuzzleBoardCube`](puzzleboard-cube.md) | cube, six PuzzleBoard faces | `pyCamSet[puzzle]` | PuzzleBoard density on Ccube geometry |

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
