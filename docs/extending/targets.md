# Extending: calibration targets

[`AbstractTarget`][pyCamSet.calibration_targets.core.abstract_target.AbstractTarget]
is the base class every calibration target extends. An implementation defines
two things: **where the feature points are** in object coordinates, and **how
to find them** in an image. Optionally it also defines a printable net and a
way of drawing itself.

`ChArUco` is the smallest real example, and wrapping OpenCV's own board is what
lets it be used to jointly calibrate *n* cameras — which the OpenCV framework
alone cannot do.

```python
class ChArUco(AbstractTarget):
```

---

## 1. Say where the points are

The first task is the `__init__`, which builds the array of feature locations:

```python exec="true"
import inspect, textwrap

from pyCamSet import ChArUco

print("```python")
print(textwrap.dedent(inspect.getsource(ChArUco.__init__)))
print("```")
```

There are three parts to it. The call to the super initialisation passes the
input arguments up, where they are stored as the reference from which the
target can be regenerated. The middle section builds the ChArUco board, keeps
it as a member so detection can use it later, and defines the `point_data`
array. The final `_process_data()` generates a local coordinate system for each
face of the target — one, here — which the initial per-camera calibrations use.

## 2. Say how to find them

With the geometry defined, `find_in_image` is what reads a photograph of it
back. It must return an
[`ImageDetection`][pyCamSet.calibration_targets.core.target_detections.ImageDetection]:

```python exec="true"
import inspect, textwrap

from pyCamSet import ChArUco

print("```python")
print(textwrap.dedent(inspect.getsource(ChArUco.find_in_image)))
print("```")
```

A standard detection, with an optional `draw` flag. With those two methods
defined the new target can be passed to `calibrate_cameras`, and used to
calibrate *n* cameras at once.

## 3. Say what your arguments mean

A target's arguments are described once, in the docstring of the constructor
that takes them, and read back out of it by `docstring_parser`. That is what
builds the GUI's forms and their validation, so a target that documents its
`:param:` entries gets a form for free and one that does not gets a bad one:

```python exec="true" result="text"
from pyCamSet import Ccube

for parameter in Ccube.construction_parameters().parameters:
    print(f"{parameter.key:<18} {parameter.label}")
```

`construction_parameters` is what decides where the corners are;
`export_parameters` is how it is drawn, which is not the same thing and is kept
separate.

## 4. Register it

Add one line to `TARGET_CLASSES`, as module and class names:

```python
TARGET_CLASSES: dict[str, tuple[str, str]] = {
    ...
    "MyTarget": ("my_package.my_target", "MyTarget"),
}
```

The name is then what the GUI offers, what a settings dictionary addresses, and
what `pyCamSet.calibration_targets.MyTarget` resolves to.
