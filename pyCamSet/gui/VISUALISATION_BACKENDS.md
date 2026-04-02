# Calibration Assessment Visualisation — Backend Issues

> **Last updated:** 2026-04-02  
> Covers `pyCamSet/utils/visualisation.py` and `pyCamSet/gui/assess_calibration.py`.

---

## 1  Open3D — EGL headless rendering is Linux-only

When the GUI attempts to render the calibration assessment offscreen into an
embedded `QLabel`, `visualise_calibration_open3d` calls Open3D's
`OffscreenRenderer`.  This renderer requires **EGL**, which is only available on
Linux (Mesa / NVIDIA EGL).  On Windows the call raises:

```
[Open3D Error] EGL Headless is not supported on this platform.
```

The existing fallback path opens a native `draw_geometries` window instead, but
that cannot deliver a pixel buffer back to the Qt widget, so the embedded image
slot remains blank.  The log entry that surfaces this is:

```
WARNING Open3D offscreen/EGL render failed (... EGL Headless is not supported
on this platform ...); retrying with interactive windowed OpenGL.
```

This is a **platform limitation of the Open3D Windows build** — EGL support is
compiled out.  There is no configuration option or driver update that enables it.

---

## 2  Open3D — Target-coordinates visualisation shows world-space points

### Symptom

The Open3D calibration view looks like a "mashup of different cubes" rather than
a single clean cube shape.  The PyVista view of the same data looks correct.

### Root cause

`visualise_calibration_open3d` is supposed to show reconstructed feature points
transformed into the **calibration-target frame** (object space).  The
transformation is computed correctly:

```python
n_inv_pose(poses[int(im)], inv_pose)
n_htform_prealloc(point, inv_pose, obj_point)   # obj_point is now in target frame
```

However, the function then stores `point` — the original **world-space**
triangulated position — rather than `obj_point`:

```python
raw_obj_points.append(point)   # ← wrong: world-space, not target-space
```

Because each image of the calibration target captures it at a different
world-space pose, all those per-image cube shapes are overlaid without
alignment, producing the observed visual noise.

The equivalent PyVista functions (`visualise_calibration` and
`render_calibration_pyvista_png`) both store `obj_point` and therefore display
correctly.

### Why the filtering still worked

`obj_point` *was* used for the outlier distance check:

```python
if np.linalg.norm(obj_point) < 3 * mean_dist:
```

So outlier rejection was correct, but the surviving points were then stored in
the wrong coordinate frame — masking the bug because the code appeared to run
without error.
