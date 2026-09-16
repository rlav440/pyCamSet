# Extending: bundle adjustment

The bundle adjustment loss is a standard loss in the computer vision
literature. In pyCamSet it is not written out as a single cost function.
Instead it is **composed**, from small pieces called function blocks, in the
style of ceres-solver.

A function block is one differentiable step in the chain that takes a point on
the calibration target all the way to a pixel. Each block declares how many
values it consumes, how many it produces, what parameters it owns, and supplies
two numba kernels: one that computes its output, and one that computes its own
derivative. Chaining the blocks gives the loss; chaining their derivatives by
the chain rule gives the analytic jacobian.

---

## Composing the loss

Blocks are combined with `+`. The standard target-pose bundle adjustment used
by `TemplateBundleHandler` is three blocks long:

```python
import pyCamSet.optimisation.function_block_implementations as fb

op_fun = fb.projection() + fb.extrinsic3D() + fb.template_points()
```

Read right to left, that is: take a point from the calibration template, move it
by the target's pose in the world, move it into the camera's frame, then project
it to pixels.

The available blocks are:

`projection`
:   Camera frame point to pixels, through the 5 term Brown–Conrady distortion.
    Owns 9 parameters per camera.

`rigidTform3d`
:   A rigid transform, 6 parameters — Rodrigues rotation and translation.

`extrinsic3D`
:   `rigidTform3d` with its parameters keyed per camera, giving the camera
    extrinsics.

`template_points`
:   `rigidTform3d` keyed per image, giving the target pose. A *template* block:
    it reads the target's feature coordinates as a given rather than
    differentiating them.

`free_point`
:   A point whose 3D location is itself optimised, for problems where the target
    geometry is not held fixed. Swapping `template_points` for this is the whole
    difference between a calibration and one that
    [frees the target's geometry](../how-to/calibrate.md#stage-5-freeing-the-targets-geometry).

`telecentric_intrinsic`
:   Camera frame point to pixels through a telecentric lens — affine, with a
    division distortion model and a residual telecentricity term. Owns 6
    parameters per camera.

`telecentric_extrinsic`
:   `rigidTform3d` keyed per camera, with **no translation**: none of the three
    components is identifiable for a telecentric camera. See
    [camera models](camera-models.md) for why, which is the more interesting
    half of the story.

Swapping a block, or adding one, changes the formulation without touching the
optimiser — this is the seam that [parameter handlers](parameters.md) and
[camera models](camera-models.md) extend through. The first two blocks of every
chain come from the cameras being calibrated, which is why the telecentric pair
substitutes for `projection() + extrinsic3D()` without anything else changing.

## What a block looks like

The projection block is the most concrete example. Its forward kernel:

```python exec="true"
import inspect, textwrap

from pyCamSet.optimisation import function_block_implementations as fb

print("```python")
print(textwrap.dedent(inspect.getsource(fb.projection.compute_fun)))
print("```")
```

Two things constrain how these are written. Both kernels are compiled with
numba, so they take only numpy arrays of a fixed shape and write their results
into a preallocated `output` buffer rather than returning them. And every block
must supply the matching `compute_jac`, whose correctness is checkable against a
numeric derivative by `abstract_function_block.test_self`.

## Code generation

The blocks are not evaluated one at a time at runtime. From the chain, pyCamSet
generates the source of a single fused loss kernel and a single fused jacobian
kernel, writes them to `pyCamSet/optimisation/template_functions/`, and imports
them back. The generated modules are named after the chain that produced them,
so a problem of a given shape is only compiled the first time it is solved and
is cached on disk thereafter.

`make_optimisation_function` drives this, and returns the pair of callables the
optimiser is handed:

```python exec="true"
import inspect, textwrap

from pyCamSet.optimisation.optimisation_handling import make_optimisation_function

print("```python")
print(textwrap.dedent(inspect.getsource(make_optimisation_function)))
print("```")
```

!!! warning "A bad jacobian is silent"

    Because the jacobian is generated source that is cached and reused, a bad
    one is otherwise reused silently — the optimiser simply converges to a worse
    answer, with no error raised. `check_jacobian_is_not_degenerate` runs on the
    jacobian actually in use, and raises when a parameter cannot reach the
    output.

## Data layout

The cost function operates over an unrolled representation of the object
geometry, of the form `(i, x, 3)`, where `i` is the number of images, `x` is the
number of unrolled keys, and 3 is the world coordinate.
