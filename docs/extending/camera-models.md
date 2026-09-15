# Extending: camera models

Most of this library assumes a pinhole camera, but that assumption is confined
to a small number of places, and replacing it is the fourth thing you can
extend. This page uses the telecentric model — shipped in
[`TelecentricCamera`][pyCamSet.cameras.telecentric_camera.TelecentricCamera] —
as a worked example, because it is about as far from a pinhole as a camera gets
and so exercises every seam.

A new camera model is four pieces:

1. A pair of **function blocks** — the projection, and whatever pose parameters
   the model can actually identify.
2. A **`Camera` subclass** that projects, undistorts and draws itself.
3. One line in **`MODEL_BLOCKS`**, tying the two together.
4. A **seed**, if OpenCV's calibration cannot produce one.

---

## What a telecentric lens is

A telecentric lens puts its aperture stop at the focal point. The chief rays are
then parallel in object space, and magnification stops depending on how far away
the object is — which is why they are the standard optic for dimensional
metrology, and why a pinhole model cannot represent one.

The full model keeps a term for the fact that no real lens is perfectly
telecentric:

$$
u = \frac{m_x\,x}{1 + \varepsilon z}\,d + c_x
\qquad
v = \frac{m_y\,y}{1 + \varepsilon z}\,d + c_y
$$

`m` is a magnification in pixels per world unit, standing where a pinhole keeps
its focal length. `d` is the division-model distortion factor. $\varepsilon$ is
the residual telecentricity error: zero for a perfect lens, in which case the
depth `z` drops out entirely and the projection is purely affine.

$\varepsilon$ is always fitted rather than switched off, so a good lens simply
returns a value near zero with a finite uncertainty.

## Start from what the geometry cannot see

This is the part that is easy to skip and expensive to skip. Before writing a
block, work out which parameters the model makes unidentifiable — because the
optimiser will not tolerate a parameter that cannot reach the residual, and it
is right not to.

A telecentric camera has **no identifiable position at all**:

- **Along its own axis.** Sliding the camera by $d$ turns
  $m x / (1 + \varepsilon(z + d))$ into
  $[m/(1+\varepsilon d)] \, x / (1 + [\varepsilon/(1+\varepsilon d)] z)$ — the
  *same* function of $(x, z)$, with a rescaled magnification and telecentricity.
  Adding the telecentricity term does not restore depth observability; it
  spreads one freedom across three parameters.
- **Across it.** Moving the camera in plane shifts every pixel by a constant,
  which the principal point absorbs. The two separate only at order
  $\varepsilon \times \mathrm{depth}$, which is not recoverable in practice.

So the in-plane position lives in $c_x, c_y$, the axial position is genuinely
unknowable, and `telecentric_extrinsic` carries **rotation only**:

```python exec="true" result="text"
from pyCamSet.optimisation import function_block_implementations as fb

for block in (fb.extrinsic3D, fb.telecentric_extrinsic):
    print(f"{block.__name__:<24} {block.params.n_params} parameters per camera")
```

Had it kept the usual six, three jacobian columns would be identically zero. The
code generator checks for exactly that (`check_all_params_reach_the_output`),
the runtime degeneracy check rechecks it, and the Schur elimination would
otherwise divide by a singular block. Three independent guards, all correct.

!!! note "This is why a telecentric rig needs more than one camera"
    A single telecentric camera cannot recover target depth at all, and a
    *planar* target under an affine camera has a two-fold out-of-plane tilt
    ambiguity: $+\theta$ and $-\theta$ image identically. Use two or more
    non-coaxial cameras, and a target with out-of-plane extent.

## The projection block

A block declares its inputs, outputs and parameters, then supplies two numba
kernels. The forward one:

```python exec="true"
import inspect, textwrap
from pyCamSet.optimisation import function_block_implementations as fb

print("```python")
print(textwrap.dedent(inspect.getsource(fb.telecentric_intrinsic.compute_fun)))
print("```")
```

Two details in there are worth stealing for any new model.

**The `1e-6` is load-bearing.** Distortion is applied to the pixel offset from
the principal point, scaled so that `k` is in units of (1000 px)⁻². Applied to
raw camera-frame coordinates instead, the squared radius would carry the
target's length unit, and `k` would land anywhere between 1e-6 and 10 depending
on whether the target was measured in millimetres or metres — small enough, at
the wrong end, to sit under the solver's step tolerance and look like a
distortion that simply refuses to fit. This is the same conditioning HALCON gets
by distorting in metric sensor coordinates.

**The division model is chosen for the direction that runs hot.**
`x_distorted = x / (1 + k r²)` makes the *forward* pass — the one the residual
evaluates millions of times — a single divide with no iteration. Inverting it is
then a quadratic root, still closed form, and used only for undistorting points.
The Brown–Conrady model in `projection` makes the opposite trade and needs a
five-iteration fixed point to undistort.

The matching `compute_jac` writes `(num_inp + n_params) * num_out` derivatives,
parameters first. Its correctness is checkable without any of the rest of the
library:

```python exec="true" result="text"
from pyCamSet.optimisation import function_block_implementations as fb

for block in (fb.telecentric_intrinsic, fb.telecentric_extrinsic):
    block().test_self()   # compares the analytic jacobian to a numeric one
    print(f"{block.__name__:<24} jacobian agrees with finite differences")
```

## The camera class

[`TelecentricCamera`][pyCamSet.cameras.telecentric_camera.TelecentricCamera]
subclasses `Camera`. More of the base class is pinhole-specific than it first
looks, and each override is forced by the geometry rather than by taste:

| Override | Why |
|---|---|
| `project_points` | affine, with no perspective divide |
| `_calc_projection_matrix` | see below |
| `undistort`, `undistort_points` | `cv2.undistort` assumes a pinhole and Brown–Conrady |
| `_make_sensormap`, `im_to_world_ray` | rays are parallel, so what varies is where each starts |
| `_update_optical_state`, `_cam_fov` | there is no focal length to read a field of view from |
| `get_mesh`, `get_viewcone` | the imaged volume is a prism, not a frustum |
| `to_param_vector`, `from_param_vector` | the handlers pack through these, so the layout lives with the model |

The projection matrix is the pleasant surprise. $u = m x/(1 + \varepsilon z) + c$
is *projective*, not merely affine, so an exact 3×4 matrix exists:

```python exec="true" result="text"
import numpy as np
from pyCamSet.cameras.telecentric_camera import TelecentricCamera
from pyCamSet.utils.general_utils import h_tform

points = np.array([[0.004, -0.003, 0.002], [-0.008, 0.006, -0.005]])

for eps in (0.0, 1.0):
    cam = TelecentricCamera(
        intrinsic=np.array([[30000.0, 0, 640.0], [0, 29000.0, 480.0], [0, 0, 1.0]]),
        res=[1280, 960], distortion_coefs=np.array([0.0]),
        telecentricity=eps, name="tc")
    error = np.max(np.abs(h_tform(points, cam.proj) - cam.project_points(points)))
    print(f"eps={eps:<5} last row of P = {cam.proj[2]}   max |P·X - project_points| = {error}")
```

A perfect lens leaves that last row `[0, 0, 0, 1]`, so the homogeneous divide is
by one and the rays meet at infinity; a real one puts the projective centre at
$z = -1/\varepsilon$. Triangulation's DLT is projective either way, so it needed
no special case — only the undistortion had to move out of the compiled kernel
and onto the camera, where each model inverts its own.

## Registering the model

One entry ties a camera class to the blocks it contributes:

```python exec="true" result="text"
from pyCamSet.optimisation.camera_models import MODEL_BLOCKS

for model, (intrinsic, extrinsic) in MODEL_BLOCKS.items():
    print(f"{model.__name__:<20} {intrinsic.__name__} + {extrinsic.__name__}")
```

The handlers read the pair off the camera set and size themselves from it, so
nothing else needs to know the parameter counts:

```python exec="true" result="text"
import pyCamSet.optimisation.function_block_implementations as fb
from pyCamSet.optimisation.camera_models import MODEL_BLOCKS

for model, (intrinsic, extrinsic) in MODEL_BLOCKS.items():
    chain = intrinsic() + extrinsic() + fb.template_points()
    widths = [b.params.n_params for b in chain.function_blocks]
    print(f"{model.__name__:<20} chain widths {widths}, "
          f"{sum(widths[:2])} parameters per camera")
```

A calibration compiles **one** kernel from **one** chain, so every camera in a
set shares a model. Mixing them is refused rather than silently mis-parameterised:

```python exec="true" result="text"
import numpy as np
from pyCamSet import CameraSet
from pyCamSet.cameras.camera import Camera
from pyCamSet.cameras.telecentric_camera import TelecentricCamera
from pyCamSet.optimisation.camera_models import blocks_for_camset

mixed = CameraSet(camera_dict={
    "cam_0": TelecentricCamera(res=[1280, 960], name="cam_0"),
    "cam_1": Camera(res=[1280, 960], name="cam_1"),
})
try:
    blocks_for_camset(mixed)
except ValueError as exc:
    print(exc)
```

Calibrate each model's cameras as their own set, then register the results.

## Seeding, when OpenCV cannot

`cv2.calibrateCamera` and `cv2.solvePnP` both assume a perspective camera, and
`solvePnP` additionally filters its solutions on the sign of a depth a
telecentric camera does not have. So a new model usually needs its own seed.

Here the seed is easy, because with distortion and telecentricity error set
aside the model is *linear* — $uv = A P + b$ — and the bundle adjustment refines
both afterwards. `pyCamSet.cameras.telecentric_calibration` is the whole thing:
a least-squares fit per view, the magnification read off as the row norms of
`A` (the rows of a rotation are unit length), and the nearest orthonormal rows
recovered by SVD.

```python exec="true" result="text"
import numpy as np
from pyCamSet.cameras.telecentric_camera import TelecentricCamera
from pyCamSet.cameras.telecentric_calibration import calibrate_telecentric
from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform

truth = TelecentricCamera(
    intrinsic=np.array([[30000.0, 0, 640.0], [0, 29000.0, 480.0], [0, 0, 1.0]]),
    res=[1280, 960], distortion_coefs=np.array([0.0]),
    telecentricity=0.0, name="tc")

rng = np.random.default_rng(0)
points = rng.uniform(-0.01, 0.01, (40, 3))          # a non-planar target
poses = [make_4x4h_tform(rng.uniform(-0.3, 0.3, 3), rng.uniform(-0.005, 0.005, 3))
         for _ in range(6)]
views = [truth.project_points(h_tform(points, pose)) for pose in poses]

magnification, principal, _, rms = calibrate_telecentric(
    [points] * len(poses), views, truth.res)
print("recovered magnification:", np.round(magnification, 3))
print("true magnification     :", truth.magnification)
print("worst view rms (px)    :", f"{np.max(rms):.2e}")
```

The principal point is pinned to the image centre rather than fitted: it is
degenerate with the in-plane translation, and only the distortion centre
separates them — which this seed does not estimate. The bundle adjustment
refines it once distortion is in the model.

Once a model has a seed, `calibrate_cameras` takes it by name:

```python
cams = calibrate_cameras(folder, target, model="telecentric")
```

## Drawing it

A pinhole camera's view frustum grows with distance, so how far to draw it is
purely a display choice. A telecentric camera's imaged cross-section is fixed by
the optics, so only the *depth* is free — and the calibration measures one:
magnification stays within a tolerance out to $|z| \le \mathrm{tol}/\varepsilon$.

```python exec="true" result="text"
import numpy as np
from pyCamSet.cameras.telecentric_camera import TelecentricCamera

K = np.array([[30000.0, 0, 640.0], [0, 29000.0, 480.0], [0, 0, 1.0]])
for eps in (2.0, 1.0, 0.0):
    cam = TelecentricCamera(intrinsic=K, res=[1280, 960], telecentricity=eps, name="tc")
    print(f"eps={eps:<5} field {np.round(cam.field_size * 1000, 2)} mm"
          f"   depth {cam.view_depth * 1000:7.1f} mm")
```

A perfect lens has no such bound, so the drawn depth is capped at
`VIEW_DEPTH_CAP_FIELDS` field widths; a box at the cap means "at least this
deep". Set `view_depth` to describe a real working volume instead. The prism is
centred on the camera frame origin rather than starting there, because that
origin is a label, not a lens position.

---

## The checklist

For a model of your own:

- [ ] Work out which parameters the geometry makes unidentifiable, and leave
      them out of the blocks. Do this first.
- [ ] Write `compute_fun` and `compute_jac`. Keep the bodies flat — the code
      generator splices their source, so no early `return`, no closures, no
      local `def`s — and check them with `test_self`.
- [ ] Pick a distortion parameterisation whose coefficients are O(1) in the
      units your targets are measured in, and whose cheap direction is the one
      the residual evaluates.
- [ ] Subclass `Camera`, and override every method that reads a focal length.
- [ ] Add the model to `MODEL_BLOCKS`.
- [ ] Provide a seed if OpenCV cannot.
- [ ] Assert recovery of *known parameters* on synthetic data, not just
      reprojection error — a wrong model can be self-consistent.

`tests/test_telecentric_model.py` is the worked version of that last point.
