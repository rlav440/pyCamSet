# Working with cameras

The [`Camera`][pyCamSet.cameras.camera.Camera] is the base component of the
library: a virtual instance standing in for a real camera. Cameras implement
the pinhole model, with a five component Brown–Conrady model of distortion.

Like a pinhole camera, a `Camera` is represented by an intrinsic matrix, an
extrinsic matrix, and a vector of distortion coefficients. It also carries the
non-mathematical things worth knowing about a camera — a name, and the
resolution of the camera it represents.

---

## Making one

A camera at the origin, with a focal length of 1000 pixels, a resolution of
640×480, and a principal point offset of [320, 240]:

```python exec="true" source="above" result="text" session="cameras"
import numpy as np

from pyCamSet import Camera

extr = np.eye(4)
intr = np.array([
    [1000,    0, 320],
    [   0, 1000, 240],
    [   0,    0,   1],
])

my_camera = Camera(extrinsic=extr, intrinsic=intr, res=(640, 480))

# a point one metre down the optical axis lands on the principal point
print(my_camera.project_points(np.array([[0.0, 0.0, 1.0]])))
```

The extrinsic transform uses 4×4 homogeneous coordinates.

## Drawing one

A camera can be drawn in a 3D plot, given the coordinates that define it.
`get_mesh` returns a PyVista mesh, which can then be plotted:

```python exec="true" source="above" session="cameras"
camera_mesh = my_camera.get_mesh()
camera_mesh.plot()
```

## Projecting through one

The primary purpose of a camera is to define the projective mapping from 3D to
2D. A single camera projects with `project_points`; to send a point through
*every* camera at once, see
[Working with camera sets](camera-sets.md#projection).

---

For everything a `Camera` carries and every method it has, see the
[API reference](../api/cameras.md).
