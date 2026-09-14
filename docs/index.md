# pyCamSet

pyCamSet is a Python library for multi-camera calibration, for MVS systems and
for instrumentation. It makes it easy to work with and to calibrate cameras,
and specifically it aims to allow **arbitrary calibration target geometries**
to be used when calibrating *n* cameras at once.

OpenCV provides methods for calibrating and working with cameras, many of which
are used here. pyCamSet adds the abstractions and the visualisation tools for
sets of calibrated cameras and for calibration targets, and an implementation
of *n* camera calibration that is easy to use.

---

## Standard components

At the heart of the library are the [`Camera`][pyCamSet.cameras.camera.Camera]
and [`CameraSet`][pyCamSet.cameras.camera_set.CameraSet] objects, and the
[`calibrate_cameras`][pyCamSet.calibration.camera_calibrator.calibrate_cameras]
function, which builds both from calibration images. A `CameraSet` saves to a
JSON formatted file, conventionally with a `.camset` extension, and records the
calibration that produced it.

The calibration process itself is defined by two extensible pieces: an abstract
model of the calibration target, and an extensible representation of the
standard bundle adjustment. Both are described under
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

## A calibration, end to end

The whole job is one call. Point it at a folder of images, one sub-folder per
camera, and give it the target that was photographed:

```python
from pyCamSet import Ccube, calibrate_cameras

cams = calibrate_cameras("path/to/images", Ccube(n_points=10, length=40))
```

It prints what it found and what it solved as it goes, and returns a
`CameraSet` you can project with, triangulate with, plot, and save.
[Calibrating a camera set](how-to/calibrate.md) walks through that output line
by line; [Self-calibration](how-to/self-calibration.md) is what to do when the
printed target is not quite the shape it was drawn as.

---

## Where to go next

| | |
|---|---|
| [Architecture](architecture.md) | How the pieces fit together |
| [Calibrating a camera set](how-to/calibrate.md) | The main path, and how to read what it tells you |
| [Self-calibration](how-to/self-calibration.md) | Letting the target's own geometry move |
| [Calibration Targets](targets/index.md) | The four targets, and how to print one |
| [The graphical workflow](how-to/gui.md) | `pycamset`, for doing the same thing with a window |
| [Extending pyCamSet](extending/index.md) | New targets, new parameters, new loss functions |
| [Troubleshooting](troubleshooting.md) | When an import fails or a target is `None` |
