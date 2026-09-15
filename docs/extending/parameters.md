# Extending: parameter handlers

A parameter handler turns a flat array of numbers into the matrices the bundle
adjustment loss is evaluated over. The standard
[`TemplateBundleHandler`][pyCamSet.optimisation.template_handler.TemplateBundleHandler]
handles the transformation for a standard, target-pose based bundle adjustment.

Extending it is how you add parameters to a calibration, or constrain the ones
that are already there. The parent class provides two hooks:

- **`add_extra_params`** — declare additional parameters, and give them an
  initial estimate
- **`parse_extra_params_and_setup`** — read those parameters back, and change
  the setup of the problem to reflect them

---

## A worked example: two rigidly fixed targets

The example below calibrates with *two* targets, held rigidly together by a
transform that is itself unknown and solved for. It is
`examples/extend_param_handler.py` in full:

```python
--8<-- "examples/extend_param_handler.py"
```

### `__init__`

The init performs a standard initialisation for one of the targets, through
`super().__init__`, which sets up the pose, extrinsic, distortion and intrinsic
automation. The data for the second target is stored alongside.

To populate the single `point_data` array both targets' arrays are flattened
and concatenated — `len0` records where the first ends, because the split has
to be undone later.

### `add_extra_params`

This adds one extra set of parameters to the optimisation: the relative pose of
the two targets, six numbers as a Rodrigues rotation and a translation. A quick
first guess at the relative transformation is taken here, because a bundle
adjustment started from an arbitrary relative pose is unlikely to find its way
to the right one.

### `parse_extra_params_and_setup`

This mutates the internal `point_data` structure to reflect the estimated
constant transformation between the targets.

!!! warning "This runs on every evaluation"

    `parse_extra_params_and_setup` is called every time the cost function is
    evaluated. Expensive operations here slow the whole optimisation down in
    proportion to the number of evaluations the solver takes — which is why the
    example writes in place, through `n_e4x4_flat_INPLACE` and
    `n_htform_broadcast_prealloc`, rather than allocating.

### `get_detection_data`

One further override is needed for this particular class. Because the target
has two detections, the data from both has to be returned. The example leans on
the superclass, calls it twice, and concatenates — applying a constant index
offset to the second detection so that its keys address the second half of the
concatenated `point_data`.

---

## Fixing the gauge

Any bundle adjustment has gauge freedoms: transformations of the whole scene
that change no reprojection at all. A handler has to remove them, or the solver
wanders along them.

`TemplateBundleHandler` fixes a camera and a pose.
[`SelfBundleHandler`][pyCamSet.optimisation.standard_bundle_handler.SelfBundleHandler],
which additionally frees the target's own points, has seven more to remove and
does it by holding seven parameters of three non-colinear target points — see
[freeing the target's geometry](../how-to/calibrate.md#stage-5-freeing-the-targets-geometry).

If your extension adds parameters that introduce a new freedom, it has to
remove it too.
