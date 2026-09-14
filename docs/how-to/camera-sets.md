# Working with camera sets

A [`CameraSet`][pyCamSet.cameras.camera_set.CameraSet] contains multiple
cameras. It is typically the output of a calibration, and so represents the
pinhole model of the system that was calibrated. Each `CameraSet` records the
calibration that created it.

A `CameraSet` is iterable, and indexable by either camera name or index number.

It is also saveable. A saved `CameraSet` is written to a JSON formatted file
for portability, by convention with a `.camset` extension. Most of the data is
directly parsable out of the JSON structure; the data and results used to
generate the calibration are compressed before being written, and all human
readable content is written to the first section of the file.

---

## Assembling one

A camera set can be assembled from lists of camera components. This builds a
synthetic ring of cameras, each defined by a rotation about the origin:

```python exec="true" source="above" session="sets"
import numpy as np

from pyCamSet import Camera, CameraSet
from pyCamSet.utils.general_utils import make_4x4h_tform

def make_cams(nc):
    # make_4x4h_tform uses the OpenCV definition of the rotation vector
    tforms = [
        make_4x4h_tform((0, b / nc * 2 * np.pi, 0), (0, 0, 0.2)) for b in range(nc)
    ]
    cams = {f"cam_{i}": Camera(extrinsic=t) for i, t in enumerate(tforms)}
    return CameraSet(camera_dict=cams)

ring = make_cams(5)
ring.plot(cam_labels=False)
```

The same code is in `examples/make_camera_ring.py`.

## Loading a saved one

Calibrating a camera set writes a saved version of the calibration into the
input directory by default. That file can be read by any other script:

```python
from pathlib import Path

from pyCamSet import load_CameraSet

my_cams = load_CameraSet(Path("my/calibration/path/optimised_cameras.camset"))
```

## Projection

A `CameraSet` provides a convenience function for projecting a point in 3D
space into all of its cameras at once:

```python exec="true" source="above" result="text" session="sets"
projection = ring.project_points_to_all_cams(np.array([0.01, 0.03, -0.05]))
print(projection)
```

## Triangulation

The inverse of projection is triangulation. If a point is seen by multiple
cameras, its location is constrained. pyCamSet performs a least squares
minimisation (DLT), which is limited in accuracy:

```python exec="true" source="above" result="text" session="sets"
recovered = ring.multi_cam_triangulate(projection)
print(recovered)
```

The input for a single feature is a dictionary whose key–value pairs are the
name of the identifying camera and the location that feature was seen at.
Multiple features can be passed as a list of such dictionaries. For rapid
triangulation of larger detection sets, the numpy array produced by
`TargetDetection.get_data()` is also accepted.

## Visualising a calibration

The quality of a calibration is the thing worth checking first. Calibration
results are compressed and saved with each `CameraSet`, so for any set that
came out of a calibration:

```python
my_cams.visualise_calibration()
```

That is the per point, per camera and per image detail underneath the numbers
in the calibration report — see
[Reading the result](calibrate.md#reading-the-result), which draws it for a
real calibration.

## Writing one out for another tool

Many external tools want a specific camera format. A calibration can be written
to COLMAP `sparse/0` format, for photogrammetry and neural reconstruction
tools:

```python
from pyCamSet.utils.saving import camset_to_colmap

camset_to_colmap(my_cams, "output/sparse/0")
```

A camera set can also be written as MVSNet formatted files with
`write_to_txt`.
