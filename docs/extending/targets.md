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

A target's arguments are described **once**, in the docstring of the
constructor that takes them. The signature already says what an argument is
called and what it defaults to, the annotation says what it holds, and the
`:param:` entry says what it means — so
[`DocumentedParameters`][pyCamSet.calibration_targets.core.parameters.DocumentedParameters]
reads all of that back out rather than having it written a second time beside
the constructor and left to drift apart.

That is what builds the GUI's forms, what a study sweeps, and what a saved spec
is checked against. A target that documents its arguments gets a form for free;
one that does not gets a `ValueError` at import.

### The grammar of a `:param:` entry

One entry carries three things, written as a sentence and a half:

```python
:param square_size: Square size (mm) -- the printed edge length of
    one chessboard square, in millimetres. Suggested: 4-40.
```

| Part | Becomes |
|---|---|
| before ` -- ` | the **label**, the name the form puts beside the box |
| after ` -- ` | the **concept**, what the form says about it |
| after `Suggested:` | the **values worth trying**, optional |

Which is exactly what comes back out:

```python exec="true" result="text"
from pyCamSet import ChArUco

for p in ChArUco.construction_parameters().parameters:
    print(f"{p.key:<16} {p.dtype:<6} default={str(p.default):<10} {p.label}")
    print(f"{'':<16} {p.concept}")
    if p.suggested:
        print(f"{'':<16} suggested: {p.suggested}")
    print()
```

### What it refuses

`parameters_from_docstring` raises rather than producing a bad form, for an
argument that:

- **the constructor does not take** — a form offering it would build a target
  nobody asked for
- **has no default** — a form starts at the default and a study samples around
  it
- **is undocumented** — what a form says about an argument is what the
  constructor says about it
- **is not annotated `int`, `float`, `bool` or `str`** — a form types one of
  those into a box
- **gives no label** — there is nothing to put beside the box

### What is deliberately *not* there

**Ranges.** No minimum, no maximum, no step. A target is an object someone
made, and a size it cannot be is one its own constructor refuses — not a range
a spin box was given. This is the difference between a form that silently
narrows `n_points=999` to something it can hold, and one that carries the value
through to the target, which then says why it cannot be that:

```python exec="true" result="text"
from pyCamSet import Ccube

try:
    Ccube(n_points=999, length=40)
except ValueError as refusal:
    print(refusal)
```

`AbstractTarget.__init__` runs that check for you, on the way in, before
anything is built from the arguments — OpenCV, for one, does not refuse a board
too small to exist; it corrupts its own state instead.

**Choices.** The one thing a docstring cannot say is a set of named values that
another library owns — a marker alphabet, say. Those are passed in beside the
arguments:

```python
return DocumentedParameters(
    cls.__init__,
    "num_squares_x", "num_squares_y", "square_size", "marker_fraction",
    "a_dict", "legacy",
    choices={"a_dict": exclude_by_prefix(
        dict_names_for_backend(ARUCO1_BACKEND), "DICT_APRILTAG_")},
)
```

That is ChArUco's list: fixed, whatever `backend` says, because the detector is
chosen in the detection phase and a board prints the same under either one.

### Two parameterisations, not one

`construction_parameters` is what decides where the corners are.
`export_parameters` is how the target is drawn — border width, DPI — which is
not what the target *is*, and is kept separate so that changing it does not
make a target something else. `detector_parameterisation` is the third: what
the detector reading it can be told, which is neither.

## 4. Register it

Add one line to `TARGET_CLASSES`, as module and class names:

```python
TARGET_CLASSES: dict[str, tuple[str, str]] = {
    ...
    "MyTarget": ("my_package.my_target", "MyTarget"),
}
```

The name is then what a settings dictionary addresses and stores, what the GUI
offers, and what `pyCamSet.calibration_targets.MyTarget` resolves to. To show the
GUI a different name, add an entry to `TARGET_LABELS` beside it â€” ChArUco is
offered as "ChArUco1" that way â€” while specs keep the registry name.
