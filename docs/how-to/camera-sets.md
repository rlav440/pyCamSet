# Working with camera sets

A [`CameraSet`][pyCamSet.cameras.camera_set.CameraSet] is three things at once,
and most of what it offers falls out of which of the three you are using:

- **A container of named cameras.** It indexes, slices, iterates and merges, and
  a slice of it is another `CameraSet`.
- **A rigid geometric object.** The cameras share one world frame, which makes
  projection into all of them and triangulation back out of them single calls,
  and makes rebasing that frame a single call as well.
- **The record of a calibration.** A set that came out of `calibrate_cameras`
  carries the detections, the residuals and the report that produced it, and
  keeps them across a save and a reload.

It is usually the output of a calibration, and so represents the pinhole model
of the system that was calibrated. It saves to a JSON formatted file, by
convention with a `.camset` extension.

---

## Making one

Two constructor forms. Either give five parallel lists — all five, or none:

```python exec="true" source="above" result="text" session="sets"
import numpy as np

from pyCamSet import Camera, CameraSet

intrinsic = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])

stereo = CameraSet(
    camera_names=["left", "right"],
    extrinsic_matrices=[np.eye(4), np.eye(4)],
    intrinsic_matrices=[intrinsic, intrinsic],
    distortion_coefs=[np.zeros(5)] * 2,
    res=[(640, 480)] * 2,
)
print(stereo.get_names())
```

or hand it a dictionary of `Camera` objects you have already built.

<div class="scene-half" markdown>

```python exec="true" source="above" session="sets"
from pyCamSet.utils.general_utils import make_4x4h_tform

def make_ring(nc):
    # make_4x4h_tform uses the OpenCV definition of the rotation vector
    tforms = [
        make_4x4h_tform((0, b / nc * 2 * np.pi, 0), (0, 0, 0.2)) for b in range(nc)
    ]
    cams = {
        f"cam_{i}": Camera(extrinsic=t, name=f"cam_{i}") for i, t in enumerate(tforms)
    }
    return CameraSet(camera_dict=cams)

ring = make_ring(5)
ring.plot(cam_labels=False)
```

</div>

The same code is in `examples/make_camera_ring.py`.

## The container

Indexing by name or by position gives a `Camera`; indexing by a slice, a list, or
an integer array gives a `CameraSet`:

```python exec="true" source="above" result="text" session="sets"
print(len(ring))
print(ring["cam_0"].position.round(3))       # by name
print(ring[-1].name)                         # by position
print(ring[1:3].get_names())                 # a slice is a CameraSet
print(ring[[0, 2, 4]].get_names())           # so is a list of indices
```

Three accessors hand out the underlying storage —
[`get_names`][pyCamSet.cameras.camera_set.CameraSet.get_names],
[`get_cam_list`][pyCamSet.cameras.camera_set.CameraSet.get_cam_list] and
[`get_cam_dict`][pyCamSet.cameras.camera_set.CameraSet.get_cam_dict] — and all
three agree on insertion order. Iteration is over the cameras themselves, and
nested iteration works, which is what makes a pairwise walk over a rig.

```python exec="true" source="above" result="text" session="sets"
baselines = {
    (a.name, b.name): np.linalg.norm(a.position - b.position)
    for a in ring for b in ring if a.name < b.name
}
print(f"{len(baselines)} pairs, widest {max(baselines.values()):.3f} m")
```

A set whose keys are integers indexes as a mapping rather than by position:
`ring[5]` finds the camera keyed `5` if there is one, and falls back to the
sixth camera if there is not.

[`make_subset`][pyCamSet.cameras.camera_set.CameraSet.make_subset] is the same
slicing with a name filter, for pulling one rig out of a set that holds several:

```python
witness = big_set.make_subset(slice(None), cam_key="witness")
```

!!! note "A subset is a view, not a copy"

    `ring[[0, 2]]` returns a new `CameraSet` holding *the same* `Camera`
    objects. Transforming the subset therefore moves those cameras in the
    parent too. `deepcopy` the subset if you want an independent one.

Two sets combine with `+`, which refuses to merge sets that share a camera name
rather than silently dropping one. Note that it mutates its left operand and
returns it, so `deepcopy` first if the original needs to survive. Equality
compares camera names as a set, then the cameras of each name against each
other, so ordering does not matter.

## Where the cameras are

The world frame is whatever the calibration chose: for a target-based
calibration it is set by the target's pose in the image the solve initialised
from, which is arbitrary. Two calls put it somewhere deliberate.

[`transform`][pyCamSet.cameras.camera_set.CameraSet.transform] post-multiplies
every extrinsic by a 4×4 homogeneous matrix. It moves the *frame*, not the
cameras within it: a point that was at `p` is at `inv(T) @ p` afterwards, and
every image is unchanged. Pass `in_place=False` for a transformed copy, leaving
the original alone.

[`set_reference_cam`][pyCamSet.cameras.camera_set.CameraSet.set_reference_cam]
is the useful special case — the frame rebased onto one camera, which is what
you want before handing the geometry to a tool that expects a reference view,
or when comparing two calibrations of the same rig:

```python exec="true" source="above" result="text" session="sets"
rebased = ring.transform(np.linalg.inv(ring["cam_2"].extrinsic), in_place=False)
rebased.set_reference_cam("cam_2")   # the same thing, in one call

print(np.allclose(rebased["cam_2"].extrinsic, np.eye(4)))
print(np.array([cam.position for cam in rebased]).round(3))
```

Two more calls change what the cameras claim about their sensors rather than
where they are.
[`scale_set_2n`][pyCamSet.cameras.camera_set.CameraSet.scale_set_2n] rescales
every camera by a power of two, for working at a reduced resolution — `1` halves
it. [`set_resolutions_from_file`][pyCamSet.cameras.camera_set.CameraSet.set_resolutions_from_file]
reads the resolutions off a folder of images laid out one sub-folder per camera,
and requires the sub-folder names to match the camera names exactly.

## Projecting into every camera

[`project_points_to_all_cams`][pyCamSet.cameras.camera_set.CameraSet.project_points_to_all_cams]
sends world points through every camera at once. A single `(3,)` point returns
one camera-name-to-coordinate dictionary; an `(n, 3)` array returns a list of
`n` of them:

```python exec="true" source="above" result="text" session="sets"
projection = ring.project_points_to_all_cams(np.array([0.01, 0.03, -0.05]))
for name, uv in projection.items():
    print(f"{name}: {uv.round(1)}")
```

Coordinates are OpenCV `(u, v)`, distorted by each camera's own coefficients
unless you pass `distort=False`.

!!! warning "Projection does not check visibility"

    Every camera in the set returns a coordinate, including cameras the point
    is behind and cameras whose sensor it misses. The projection is the pinhole
    equation, not a render.
    [`Camera.can_image`][pyCamSet.cameras.camera.Camera.can_image] is the
    bounds check, per camera:

    ```python exec="true" source="above" result="text" session="sets"
    edge = np.array([0.15, 0.0, 0.15])      # out towards the rim of the ring

    print(len(ring.project_points_to_all_cams(edge)), "coordinates returned")
    print([cam.name for cam in ring if cam.can_image(edge)], "can actually see it")
    ```

    Synthetic data built without that check is the usual cause of a
    triangulation that is worse than it should be: three of those five
    coordinates are off the sensor, and the least squares fit weights them the
    same as the two that are real.

## Triangulating back out

The inverse of projection. Given a feature seen by two or more cameras, its
world position is constrained, and
[`multi_cam_triangulate`][pyCamSet.cameras.camera_set.CameraSet.multi_cam_triangulate]
recovers it by least squares (DLT) over the observations:

```python exec="true" source="above" result="text" session="sets"
print(ring.multi_cam_triangulate(projection))
```

It accepts three input shapes, which is the reason it reads oddly at first:

| Input | Meaning |
|---|---|
| one dict | a single feature: `{camera name: (u, v)}` |
| a list of dicts | one entry per feature |
| an `(n, 5)` array | the raw form, as `TargetDetection.get_data()` returns it — `[camera index, image number, key, u, v]` |

The array form is the one to use for a whole detection set; building thousands
of dictionaries to hand it a detection it already has in array form is the slow
way round.

!!! warning "Features seen by fewer than two cameras are dropped"

    The routine finds shared visibility itself, and a feature only one camera
    saw cannot be triangulated, so it is silently discarded. **The returned
    array is therefore shorter than the input, and its rows do not line up with
    your features:**

    ```python exec="true" source="above" result="text" session="sets"
    points = np.array([[0.01, 0.03, -0.05], [0.0, 0.01, 0.0], [0.02, -0.01, 0.03]])
    seen = ring.project_points_to_all_cams(points)
    seen[1] = {"cam_1": seen[1]["cam_1"]}     # the middle point, seen once

    print(ring.multi_cam_triangulate(seen).round(3))
    ```

    Pass `return_used=True` to recover the correspondence. It returns four
    things — the points, the observations that survived, the row indices each
    point was built from, and the identifying key of each point:

    ```python exec="true" source="above" result="text" session="sets"
    recovered, used_rows, from_rows, keys = ring.multi_cam_triangulate(
        seen, return_used=True)

    print(keys)                                    # which features came back
    print([len(rows) for rows in from_rows])       # how many views built each
    ```

    `keys` is the `[image number, key]` column pair of each returned point, so
    it is what tells you that the middle feature is missing. `from_rows` indexes
    into `used_rows`, which is how a per-observation quantity — a reprojection
    residual, say — gets averaged per triangulated point. The front page's
    scene does exactly that.

## Measuring with calibrated cameras.

Once the cameras are calibrated, the target is a well characterised fiducial
rather than a calibration object.
[`find_target_pose`][pyCamSet.cameras.camera_set.CameraSet.find_target_pose]
holds every camera at its calibration and solves only the target's pose, so
this is a measurement with the rig rather than a recalibration of it:

```python
pose = cameras.find_target_pose({name: images[name] for name in cameras.get_names()}, target)
poses = cameras.find_target_poses(image_sequences, target)
```

`find_target_poses` takes one list of images per camera, indexed by timestep, and
returns an `(n_timesteps, 4, 4)` array. See
[Choosing a target](../targets/index.md) for which targets suit this.

## Drawing a camera set.

[`plot`][pyCamSet.cameras.camera_set.CameraSet.plot] opens the set in a 3D
window, with the world origin drawn as an RGB axis triad. It takes
`additional_mesh` — a PyVista mesh, a raw `(n, 3)` array, another `CameraSet`,
or a list mixing them — so a reconstruction can be drawn against the cameras
that produced it. `view_cones=True` adds each camera's field of view, and
[`plot_np_array`][pyCamSet.cameras.camera_set.CameraSet.plot_np_array] is the
shorthand for the array case.

[`get_scene`][pyCamSet.cameras.camera_set.CameraSet.get_scene] returns the same
scene as a PyVista `Plotter` without showing it, which is the one to reach for
when you want to compose:

```python exec="true" source="above" session="sets"
import pyvista as pv

points = ring.multi_cam_triangulate(
    ring.project_points_to_all_cams(np.random.default_rng(0).normal(0, 0.02, (300, 3)))
)

scene = ring.get_scene(labels=False)
scene.add_mesh(pv.PolyData(points), color="r", point_size=4,
               render_points_as_spheres=True)
scene.show()
```

[`get_camera_meshes`][pyCamSet.cameras.camera_set.CameraSet.get_camera_meshes]
gives the frustum meshes alone, for building a scene from scratch. All of these
need PyVista, which the default install includes.

[`draw_camera_distortions`][pyCamSet.cameras.camera_set.CameraSet.draw_camera_distortions]
is the one 2D view: a quiver plot per camera of where its distortion model moves
a pixel. It, like `visualise_calibration` below, takes `show=False` with a
`save_dir` to write the figures instead of opening windows, which is what a run
with nobody watching it wants.

## What a calibration leaves behind

A set that came out of a calibration carries it, and this is the first thing to
look at before trusting anything above.
[`calibration_report`][pyCamSet.utils.calibration_report.CalibrationReport] is
the numbers — error distribution, per-camera table, flags raised by the run:

```python
print(cams.calibration_report.summary())
```

[`visualise_calibration`][pyCamSet.cameras.camera_set.CameraSet.visualise_calibration]
draws all five diagnostic plots underneath those numbers, and
[`calibration_diagnostics`][pyCamSet.cameras.camera_set.CameraSet.calibration_diagnostics]
returns the shared work they are drawn from, so that one can be drawn on its
own or two calibrations compared:

```python
cams.visualise_calibration()
cams.visualise_calibration(show=False, save_dir="figures/")
```

[The error distribution](calibrate.md#the-error-distribution) works through
every one of those plots on a real calibration, and is where to read about what
they mean. The raw material is on the set as `calibration_params`,
`calibration_result` (the residual the solver stopped at), `calibration_jac` and
`calibration_handler`, and `get_calibration_points()` is the triangulated
detections with the gross outliers removed.

## Saving and loading

```python exec="true" source="above" result="text" session="sets"
from pathlib import Path
from tempfile import mkdtemp

from pyCamSet import load_CameraSet

path = Path(mkdtemp()) / "ring.camset"
ring.save(path)

print(load_CameraSet(path) == ring)
```

Calibrating writes `optimised_cameras.camset` into the image directory by
default, so the usual read is of a file you did not explicitly save.

The file is JSON. The cameras, the target's constructor arguments, the fixed
parameters and the report are all plain readable structure at the top of it; the
detections, the residuals and the Jacobian are compressed blobs after them,
because they are large and nobody reads them by eye.

A reload rebuilds the detections, the target and the parameter handler as well
as the cameras, which is why `visualise_calibration` and
`calibration_diagnostics` still work on a set that came off disk months later.
If any of those cannot be rebuilt — a target whose class has since changed its
arguments, say — loading logs a warning and returns the cameras alone rather
than failing, so a set that loads quietly but has no diagnostics is a set whose
warning you missed.

## Handing it to another tool

External tools each want their own format. COLMAP, for photogrammetry and
neural reconstruction:

```python
from pyCamSet.utils.saving import camset_to_colmap

camset_to_colmap(cams, "output/sparse/0")
```

which writes `cameras.txt` and a `rig_config.json` carrying the inter-camera
geometry, so COLMAP can hold the rig rigid. `ref_cam_name` chooses the reference
camera; it defaults to the first in the set. `examples/convert_to_colmap.py` is
this end to end.

MVSNet and ACMMP, for dense multi-view stereo, want per-camera files plus a
depth range and a view-pair list, which is
[`write_to_txt`][pyCamSet.cameras.camera_set.CameraSet.write_to_txt]:

```python
from pyCamSet.reconstruction.acmmp_utils import ReconParams

cams.write_to_txt(Path("output/cams"), ReconParams(mindist=0.1, maxdist=0.8), ims=images)
```

The depth range is in the units the calibration is in, and is the one thing here
that cannot be inferred from the camera set — it is a statement about the scene.
Passing `ims` also undistorts and writes the images themselves, since a depth
estimator wants them rectified.

The view-pair list, `pair.txt`, is scored for the rig it is given. A rig whose
cameras converge on a shared target — an ordinary calibration rig — is scored by
the angle each pair subtends at that convergence point; anything else, such as a
forward-facing capture, is scored by the angle between view vectors and windowed
by `ReconParams`' `minangle` and `maxangle`. Each reference view keeps its
`max_n_view` best candidates, written best first and scored relative to its own
best one. Pass `pair_scoring="view_angle"` or `pair_scoring="convergence"` to
choose instead of letting the rig decide.

---

For every method and everything a `CameraSet` carries, see the
[API reference](../api/cameras.md).
