# pyCamSet

pyCamSet is a Python library for multi-camera calibration, for MVS systems and
for instrumentation.  
It builds on OpenCV, with the aim of making it easy to use different calibration targets when calibrating *n* cameras at once.

---

## What it works with

pyCamSet uses a known calibration target to identify the unknown relationships between
*n* input camera objects.

<div class="scene-row" markdown>

<div markdown>
```python exec="true" session="front"
import numpy as np

from pyCamSet import Camera

camera = Camera(
    extrinsic=np.eye(4),
    intrinsic=np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]),
    res=(640, 480),
)
camera.get_mesh().plot(style="wireframe", color="k", window_size=(700, 700))
```

**A camera** — A frustrum representation of the pinhole camera.
The square corner of the frustrum indicates a 100x100 subwindow of the camera.
</div>

<div markdown>
```python exec="true" session="front"
from pyCamSet import Ccube

target = Ccube(n_points=10, length=40)
scene = target.plot(return_scene=True)
scene.window_size = (700, 700)
scene.show()
```

**A target** — here a Ccube: six ChArUco faces on a 40 mm cube, which the
library both draws and prints as a foldable net.
</div>

<div markdown>
```python exec="true" log="no" session="front"
import pyvista as pv
from cv2 import aruco

from pyCamSet import calibrate_cameras

cams = calibrate_cameras(
    "tests/test_data/calibration_ccube",
    Ccube(n_points=10, length=40,
          aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.2),
    threads=1,
    optimise_target=True,
)

data = cams.calibration_handler.get_detection().get_data()
residual = np.linalg.norm(cams.calibration_result.reshape(-1, 2), axis=1)

# A point is triangulated from a contiguous run of the observations of it, and
# the residuals are ordered as the detection is, so both are put in that order.
order = np.lexsort(data[:, 1:-2].T[::-1])
residual = residual[order]
points, _, seen_by, _ = cams.multi_cam_triangulate(data[order], return_used=True)

cloud = pv.PolyData(points)
cloud["Reprojection error (px)"] = [np.mean(residual[rows]) for rows in seen_by]

scene = cams.get_scene(labels=False)
scene.window_size = (700, 700)
# A quarter turn about the world z: the default isometric view puts the three
# cameras almost in line with each other, and this separates them.
scene.camera.azimuth = 90
scene.add_mesh(cloud, render_points_as_spheres=True, point_size=1,
               clim=[0, 3 * np.median(residual)],
               scalar_bar_args={"position_x": 0.15, "position_y": 0.9,
                                "width": 0.7, "height": 0.04})
scene.camera.zoom(1.15)
scene.show()
```

**A calibration** — three cameras where the solve put them, and every corner
the target showed them triangulated back into the scene, coloured by the error
it was reconstructed with.
</div>

</div>

At the heart of the library are the [`Camera`][pyCamSet.cameras.camera.Camera]
and [`CameraSet`][pyCamSet.cameras.camera_set.CameraSet] objects, and the
[`calibrate_cameras`][pyCamSet.calibration.camera_calibrator.calibrate_cameras]
function, which builds both from calibration images. A `CameraSet` saves to a
JSON formatted file, conventionally with a `.camset` extension, and records the
calibration that produced it.
The calibration process itself is defined by two extensible pieces: an abstract
model of the calibration target, and an extensible representation of the
standard bundle adjustment, 
[Extending pyCamSet](extending/index.md).

---

## Getting started

Install from PyPI:

```bash
pip install pyCamSet
```

That is the full install: the library, plotting, and the graphical calibration
workflow, which is launched with `pycamset`. The GUI toolkit is the largest
dependency by a wide margin, so for a server or a CI run there is a lean
install that leaves it out:

```bash
pip install pyCamSet --no-deps
pip install -r requirements_core.txt
```

Everything except the GUI works from the lean install, plotting included.

Or clone it, which is what enables development and extension:

```bash
git clone https://github.com/rlav440/pyCamSet
pip install -e .
```

---

## A quick example.

The whole job is one call. Point it at a folder of images, one sub-folder per
camera, and give it the target that was photographed:

```python
from pyCamSet import Ccube, calibrate_cameras

cams = calibrate_cameras(
    "path/to/images", Ccube(n_points=10, length=40), optimise_target=True)
```

`optimise_target=True` solves the shape of the target as well, in a second
bundle adjustment started from the first — a printed target is never quite the
object it was drawn as, and what it gets wrong is charged to the cameras until
this is allowed to move. It costs the claim the target made about scale.
Further detail is given in [Stage 5 of calibrating a camera set](how-to/calibrate.md#stage-5-freeing-the-targets-geometry).

It prints what it found and what it solved as it goes, and returns a
`CameraSet` you can project with, triangulate with, plot, and save.
A calibration outputs a [`calibration_report`][pyCamSet.utils.calibration_report.CalibrationReport].
This describes the residuals o 
```python exec="true" source="above" result="text" session="front"
print(cams.calibration_report.summary())
```

`initial` is the mean reprojection error the solve started from and
`final mean` what it ended at, with the per-camera table saying which camera
carries the rest.
[Calibrating a camera set](how-to/calibrate.md) gives more detail on how to use the CLI or GUI to calibrate.

---

## Where to go next

| | |
|---|---|
| [Calibrating a camera set](how-to/calibrate.md) | The main path, and how to read what it tells you |
| [Calibration Targets](targets/index.md) | The four targets, and how to print one |
| [The graphical workflow](how-to/gui.md) | `pycamset`, for doing the same thing with a window |
| [Architecture](architecture.md) | How the pieces fit together |
| [Extending pyCamSet](extending/index.md) | New targets, new parameters, new loss functions |
| [Troubleshooting](troubleshooting.md) | When an import fails or a target is `None` |
