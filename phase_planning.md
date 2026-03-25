# pyCamSet Calibration Phase Map

> **Repository:** `ColDSnit/pyCamSet` (development branch, commit `b180327`)
>
> **Date compiled:** 2026-03-13
>
> **Purpose:** A complete decomposition of the pyCamSet multi-camera calibration
> pipeline into discrete phases and sub-phases, traced directly from source code.
> This document serves as the design reference for a future `phased_calibration`
> module within `ColDSnit/pyCamSet`.
>
> **Status:** Reference document — no code changes proposed yet.
>
> **Future goals:**
> 1. Implement a `phased_calibration` subfolder in `ColDSnit/pyCamSet` that
>    exposes each phase as a separable, re-runnable step with checkpoint
>    serialisation (pickle/camset) at every phase boundary.
> 2. Attach quantitative and visual diagnostics at the end of each phase so
>    users can assess data quality and calibration progress before proceeding.
> 3. Build a GUI front-end atop the phased structure, allowing non-coding users
>    to step through calibration, inspect diagnostics, and adjust parameters at
>    each phase.

---

## Table of Contents

- [Overview](#overview)
- [Phase 0 — Input Validation & Setup](#phase-0--input-validation--setup)
- [Phase 1 — Target Detection](#phase-1--target-detection)
- [Phase 2 — Per-Camera Initial Calibration (Intrinsics)](#phase-2--per-camera-initial-calibration-intrinsics)
- [Phase 3 — Template Bundle Adjustment (Stereo Calibration)](#phase-3--template-bundle-adjustment-stereo-calibration)
- [Phase 4 — Self-Calibration with Free Target Points](#phase-4--self-calibration-with-free-target-points)
- [Phase 5 — Visualisation & Validation](#phase-5--visualisation--validation)
- [Resolved Uncertainties](#resolved-uncertainties)
- [Remaining Open Questions](#remaining-open-questions)
- [Per-Phase Diagnostic Proposals](#per-phase-diagnostic-proposals)
- [Summary Table](#summary-table)
- [Source File Index](#source-file-index)

---

## Overview

This document maps the full calibration pipeline implemented in pyCamSet into
six phases (0–5). The mapping was produced by tracing the source code of the
`development` branch of `ColDSnit/pyCamSet`, starting from the top-level entry
point `calibrate_cameras()` and continuing through the user-script-initiated
self-calibration stage.

The intended use of this document is threefold:

1. **As a reference** — so that anyone reading or modifying pyCamSet can
   understand what happens at each stage, which functions are called, what data
   flows between them, and what parameters are user-tunable.
2. **As a design specification** — for a `phased_calibration` module that
   decomposes the monolithic `calibrate_cameras()` call into discrete,
   independently re-runnable steps with checkpoint files and diagnostic outputs.
3. **As a GUI blueprint** — each phase and its diagnostics map naturally to a
   wizard-style interface where a user can inspect, adjust, and proceed.

The pipeline has two stages. Stage 1 (Phases 0–3) is handled by
`calibrate_cameras()` in `camera_calibrator.py`. Stage 2 (Phase 4) is not
called by `calibrate_cameras()`; it is a separate, user-initiated step using
`SelfBundleHandler`, as demonstrated in the attached `calibrate_ccube.py`.
Phase 5 (visualisation) is called manually after either stage.

```
Stage 1: calibrate_cameras()
  ├── Phase 0:   Input validation & setup
  ├── Phase 1:   Target detection
  │    ├── 1a: Subfolder discovery & image sanitisation
  │    ├── 1b: Per-camera corner detection
  │    └── 1c: Detection validation
  ├── Phase 2:   Per-camera initial calibration (intrinsics)
  │    ├── 2a: Best-pose image selection
  │    ├── 2b: OpenCV per-camera calibration
  │    └── 2c: High-distortion re-detection (conditional)
  └── Phase 3:   Template bundle adjustment (stereo calibration)
       ├── 3a: Handler construction
       ├── 3b: Initial parameter estimation & outlier rejection
       ├── 3c: Loss & Jacobian compilation
       └── 3d: Nonlinear least squares optimisation

Stage 2: User script (e.g. calibrate_ccube.py)
  └── Phase 4: Self-calibration with free target points
       ├── 4a: SelfBundleHandler construction
       ├── 4b: Warm-start from Phase 3
       ├── 4c: Bundle adjustment (free points)
       └── 4d: Gauge transform (inside get_camset)

Post-processing:
  └── Phase 5: Visualisation & validation
```

---

## Phase 0 — Input Validation & Setup

**Source:**
[`camera_calibrator.py` L49–66](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/calibration/camera_calibrator.py#L49-L66)

**What happens:**

1. `f_loc` is coerced from `str` to `Path` if necessary.
2. `save_loc` defaults to `f_loc` when not explicitly provided.
3. Thread count is auto-selected as `min(max(1, cpu_count()-2), 20)`.

**Data in → Data out:** Raw user parameters → validated parameters (no
serialisable artefact).

**User-tunable parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `f_loc` | `Path\|str` | Required | Root folder containing per-camera subfolders |
| `save_loc` | `Path\|None` | `f_loc` | Where to save outputs |
| `save` | `bool` | `True` | Whether to cache/save artefacts |
| `draw` | `bool` | `False` | Draw detections as they happen |
| `n_lim` | `int\|None` | `None` | Max images to use per camera |
| `threads` | `int\|None` | Auto | Number of threads for optimisation |
| `high_distortion` | `bool` | `False` | Enable iterative high-distortion scheme |
| `fixed_params` | `dict\|None` | `None` | Lock specific camera parameters |
| `problem_options` | `dict\|None` | `None` | Options passed to the optimiser |

**Phase 0 diagnostics:** Minimal. Confirm folder structure validity
(subfolders exist, images found), print resolved parameter summary.

---

## Phase 1 — Target Detection

### Sub-phase 1a — Subfolder Discovery & Image Sanitisation

**Source:**
[`camera_calibrator.py` L282–296](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/calibration/camera_calibrator.py#L282-L296)

**What happens:**

1. `get_subfolder_names(f_loc)` identifies one subfolder per camera view.
2. `sanitise_input_images()` checks that all camera folders contain the same
   number of images.
3. Cache check — if `detected_datapoints.pickle` exists and caching is enabled,
   load from cache and skip sub-phase 1b entirely.

### Sub-phase 1b — Per-Camera Corner Detection

**Source:**
[`camera_calibrator.py` L297–320](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/calibration/camera_calibrator.py#L297-L320)
→ [`abstract_target.py` L90–128](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/calibration_targets/abstract_target.py#L90-L128)

**What happens:**

1. For each camera subfolder, call `calibration_target.find_in_imfolder(...)`.
2. For each image file (naturally sorted): read via `cv2.imread` → convert to
   single-channel greyscale → call `find_in_image()`.
3. `find_in_image()` is target-specific. For `Ccube`, it uses ArUco marker
   detection followed by ChArUco corner interpolation.
4. Each call returns an `ImageDetection` (matched key–point pairs).
5. Per-camera detections are accumulated into a `TargetDetection` object.
6. All per-camera `TargetDetection` objects are merged via `reduce(+)`.
7. Camera resolutions are extracted by reading the first image per camera.
8. The merged detection + resolutions are pickled to
   `detected_datapoints.pickle`.

### Sub-phase 1c — Detection Validation

**Source:**
[`camera_calibrator.py` L330–373](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/calibration/camera_calibrator.py#L330-L373)

**What happens:**

1. For each camera, compute the fraction of images in which the board was
   detected (detection rate).
2. For each camera, compute the average board completeness (fraction of expected
   corners actually found per detection).
3. Log warnings if detection rate < 90% or completeness < 50%.

**Data in → Data out (whole Phase 1):** Image folder structure →
`TargetDetection` + `list[tuple]` of camera resolutions.

**Artefact:** `detected_datapoints.pickle`

**User-tunable parameters:**

| Parameter | Purpose |
|-----------|---------|
| `draw` | Show detection overlays per image |
| `n_lim` | Limit number of images processed |
| `caching` | Whether to read/write the pickle cache |
| Target definition (e.g. `Ccube(n_points=6, length=30)`) | Controls point layout and detection algorithm |

**Phase 1 diagnostics — see [Per-Phase Diagnostic Proposals](#per-phase-diagnostic-proposals).**

---

## Phase 2 — Per-Camera Initial Calibration (Intrinsics)

### Sub-phase 2a — Best-Pose Image Selection

**Source:**
[`camera_calibrator.py` L140–148](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/calibration/camera_calibrator.py#L140-L148)

**What happens:**

1. Build a features-per-image-per-camera matrix via
   `features_per_im_per_cam()`.
2. Pick the image with the highest total detection score where every camera has
   ≥ 6 features. This image is used for the initial extrinsic (PnP) estimate
   per camera.

### Sub-phase 2b — OpenCV Per-Camera Calibration

**Source:**
[`camera_calibrator.py` L148–169](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/calibration/camera_calibrator.py#L148-L169)
→ [`abstract_target.py` L266–347](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/calibration_targets/abstract_target.py#L266-L347)

**What happens:**

1. Cache check — if `initial_cameras.camset` exists, load and return.
2. For each camera independently:
   - Extract per-image object points (3D, from target geometry) and image points
     (2D, from detections).
   - For multi-board targets (Ccube): group detections by board ID, require
     > 12 points per board.
   - Call `cv2.calibrateCamera(object_points, image_points, res, None, None)`.
   - Construct a `Camera` object with the resulting intrinsic matrix and
     distortion coefficients.
   - Use `cv2.solvePnPGeneric` on the best-pose image to get an initial
     extrinsic estimate.
3. Assemble `CameraSet` — `{cam_name: Camera}` →
   `CameraSet(camera_dict=...)`.
4. Save to `initial_cameras.camset`.

### Sub-phase 2c — High-Distortion Re-detection (Conditional)

**Source:**
[`camera_calibrator.py` L86–106](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/calibration/camera_calibrator.py#L86-L106)

**Condition:** Only runs if `high_distortion=True`.

1. Re-run detection (`detect_datapoints_in_imfile`) passing in the `CameraSet`
   from sub-phase 2b. This allows the detector to undistort images using the
   initial intrinsics before looking for corners.
2. Re-run `run_initial_calibration` with the improved detections.
3. Call `initial_cams.draw_camera_distortions()` for visual inspection.

**Data in → Data out (whole Phase 2):** `TargetDetection` + `cam_res` →
`CameraSet` (each camera has intrinsics + distortion; extrinsics are
per-camera-only, not yet globally consistent).

**Artefact:** `initial_cameras.camset` (or
`initial_cameras_high_distortion.camset` if 2c ran).

**User-tunable parameters:**

| Parameter | Purpose |
|-----------|---------|
| `fixed_params` | Lock intrinsics (`'int'`) or extrinsics (`'ext'`) per named camera |
| `high_distortion` | Enable the 2c re-detection loop |

**Phase 2 diagnostics — see [Per-Phase Diagnostic Proposals](#per-phase-diagnostic-proposals).**

---

## Phase 3 — Template Bundle Adjustment (Stereo Calibration)

**Source:**
[`camera_calibrator.py` L210–259](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/calibration/camera_calibrator.py#L210-L259)

This phase jointly optimises all camera intrinsics, extrinsics, and target poses
using a least-squares solver. The target geometry is held **fixed**
(template-based).

### Sub-phase 3a — TemplateBundleHandler Construction

**Source:**
[`template_handler.py` L100–152](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/template_handler.py#L100-L152)

**What happens:**

1. Deep-copy the detection and target data to avoid mutation.
2. Determine which intrinsics, extrinsics, and poses are "unfixed" based on
   `fixed_params` and `problem_options`.
3. Create a `TemplateBundlePrimitive` — raw parameter storage arrays for poses
   (n_images × 6), extrinsics (n_cameras × 6), and intrinsics (n_cameras × 9).
4. Fix the reference pose (default image 0) to the identity to resolve the
   gauge ambiguity.
5. Set the cost function block chain:
   `projection() + extrinsic3D() + template_points()`.

**Default problem options** (from `DEFAULT_OPTIONS`):

| Option | Default | Purpose |
|--------|---------|---------|
| `verbosity` | `2` | Solver output verbosity |
| `fixed_pose` | `0` | Image index whose target pose is fixed to identity |
| `ref_cam` | `0` | Reference camera index |
| `ref_pose` | `0` | Reference pose index |
| `outliers` | `'ask'` | Outlier handling: `'ask'` (interactive), `'y'` (auto-remove), `'n'` (skip) |
| `max_nfev` | `100` | Maximum function evaluations for `least_squares` |

### Sub-phase 3b — Initial Parameter Estimation & Outlier Rejection

**Source:**
[`template_handler.py` L303–347](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/template_handler.py#L303-L347)
→ [`estimate_camera_relative_poses()` L468–601](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/template_handler.py#L468-L601)
→ [`find_and_exclude_transform_outliers()` L243–279](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/template_handler.py#L243-L279)

**What happens:**

1. **Per-camera, per-image PnP** — for every (camera, image) pair, call
   `target_pose_in_cam_image()` which uses `cv2.solvePnPGeneric()` to estimate
   the 4×4 target-to-camera transform. Returns `NaN` if detection is missing.
2. **Reference pose selection** — `check_feasiblity_and_update_refpose()` finds
   an image visible to **all** cameras. If the default `ref_pose` is not visible
   to all, it picks the first one that is.
3. **Relative camera poses** — from the reference pose's per-camera PnP results,
   derive camera-to-camera transforms. Then for each camera, derive all
   per-image target poses in a consistent reference frame.
4. **Best-estimate selection** — for each image, evaluate the reprojection cost
   using each camera's derived target pose, and select the one with lowest
   error.
5. **Missing pose detection** — images where PnP failed for all cameras get
   `NaN` transforms → flagged as `missing_poses`.
6. **Outlier rejection** — `find_and_exclude_transform_outliers()`:
   - Uses MAD (Median Absolute Deviation) on the per-image reprojection error
     array with `out_thresh=20`.
   - Behaviour depends on `problem_opts['outliers']`:
     - `'ask'`: interactive prompt for each outlier iteration.
     - `'y'`: auto-remove outliers.
     - `'n'`: skip outlier rejection entirely.
   - Iterates up to 10 times, removing flagged images from `missing_poses` each
     loop until no more outliers are found.
7. **Pack parameter vector** — concatenate unfixed intrinsics, extrinsics, and
   poses into a single 1D `numpy` array.

### Sub-phase 3c — Loss & Jacobian Compilation

**Source:**
[`optimisation_handling.py` L24–49](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/optimisation_handling.py#L24-L49)
→ [`abstract_function_blocks.py` L291–420](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/abstract_function_blocks.py#L291-L420)

**What happens:**

1. `param_handler.get_initial_params()` triggers sub-phase 3b if not already
   computed.
2. `param_handler.make_loss_fun(threads)` → invokes the abstract function block
   system:
   - Constructs a filename from the function block chain
     (e.g. `loss_projection_extrinsic3D_template_points.py`).
   - If no cached `.py` file exists, **dynamically generates Python source code**
     containing a `@njit(parallel=True, fastmath=True, cache=True)` numba
     function, writes it to `pyCamSet/optimisation/template_functions/`, then
     imports and compiles it.
   - If the file already exists, imports and uses the cached version directly.
   - The generated function evaluates `reprojection_error = projected_point -
     detected_point` for every observation, parallelised across threads.
3. `param_handler.make_loss_jac(threads)` → same pattern for the Jacobian:
   - If all function blocks implement `compute_jac`, generates an analytical
     Jacobian function and a chain-rule "matflow" function.
   - Returns a function that produces CSR sparse matrix data.
   - If any block lacks `compute_jac`, `can_make_jac()` returns `False` and the
     Jacobian falls back to `"2-point"` finite differences inside
     `scipy.optimize.least_squares`.

### Sub-phase 3d — Nonlinear Least Squares Optimisation

**Source:**
[`optimisation_handling.py` L51–118](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/optimisation_handling.py#L51-L118)

**What happens:**

1. Evaluate initial error using the compiled loss function.
2. Log parameter count, control point count, and initial Euclidean reprojection
   error (px).
3. Warn if initial error > 150 px or NaN.
4. Call `scipy.optimize.least_squares()` with:
   - `verbose=problem_opts['verbosity']`
   - `jac=bundle_jac` (analytical) or `"2-point"` (finite-diff fallback)
   - `max_nfev=problem_opts['max_nfev']`
   - `x_scale='jac'` (scale parameters by Jacobian column norms)
   - Solver: Trust Region Reflective (TRF). The `method="lm"` and
     `loss="cauchy"` lines are commented out in source.
5. Log final Euclidean error and wall-clock time.
6. Warn if final error > 5 px.
7. Extract the optimised `CameraSet` via
   `param_handler.get_camset(optimisation.x)`.
8. Attach calibration history:
   `camset.set_calibration_history(optimisation, param_handler)`.
9. Run a verification evaluation of the loss at the solution point.

**Data in → Data out (whole Phase 3):** `CameraSet` (initial intrinsics) +
`TargetDetection` + `AbstractTarget` → optimised `CameraSet` (intrinsics +
extrinsics + target poses jointly refined, target geometry held fixed).

**Artefact:** `optimised_cameras.camset`

**Phase 3 diagnostics — see [Per-Phase Diagnostic Proposals](#per-phase-diagnostic-proposals).**

---

## Phase 4 — Self-Calibration with Free Target Points

**Source:**
[`standard_bundle_handler.py` L123–480](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/standard_bundle_handler.py#L123-L480),
user script `calibrate_ccube.py`

This phase is **not** called by `calibrate_cameras()`. The user manually
constructs a `SelfBundleHandler` and calls `run_bundle_adjustment()`. It extends
Phase 3 by also allowing the target's 3D point positions to vary.

### Sub-phase 4a — SelfBundleHandler Construction

**Source:**
[`standard_bundle_handler.py` L128–182](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/standard_bundle_handler.py#L128-L182)

**What happens:**

1. Call `super().__init__()` (TemplateBundleHandler), inheriting all Phase 3
   setup.
2. Flatten target point data to a 1D array.
3. **Fix gauge symmetry** — find 3 non-colinear points on the target
   (`find_not_colinear_pts`), then:
   - Fix all 3 coordinates of point 0 (3 DOF).
   - Fix all 3 coordinates of point 1 (3 DOF).
   - Fix 1 coordinate of point 2 (1 DOF).
   - Total: 7 DOF fixed, breaking the 7-DOF similarity gauge (3 rotation +
     3 translation + 1 scale).
4. **Visibility masking** — check which target points were actually detected.
   Unseen points are also fixed (their coordinates are not optimised).
5. Replace the `TemplateBundlePrimitive` with a `StandardBundlePrimitive` that
   includes the free-point parameter block.
6. Set the cost function block chain:
   `projection() + extrinsic3D() + rigidTform3d() + free_point()`.

### Sub-phase 4b — Warm-Start from Phase 3

**Source:**
[`standard_bundle_handler.py` L262–278](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/standard_bundle_handler.py#L262-L278)

**What happens:**

1. Verify that the previous `CameraSet` was calibrated using a
   `TemplateBundleHandler` (raises `ValueError` otherwise).
2. Copy the Phase 3 camera parameter vector (intrinsics + extrinsics + poses)
   into the first portion of the new parameter vector.
3. Copy the target point data (flattened, only unfixed points) into the
   remaining portion.
4. Carry forward the `missing_poses` mask from Phase 3.

### Sub-phase 4c — Bundle Adjustment (Free Points)

Same `run_bundle_adjustment()` pipeline as sub-phase 3d, but now jointly
optimising camera parameters **and** target geometry. The parameter vector is
larger because it includes the unfixed target point coordinates.

### Sub-phase 4d — Gauge Transform

**Source:**
[`standard_bundle_handler.py` L340–410](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/standard_bundle_handler.py#L340-L410)

**What happens (inside `get_camset()` and `apply_gauge_transform()`):**

Because the optimisation has gauge freedom, a post-optimisation gauge correction
maps the result back to the original target's coordinate frame:

1. Compute pairwise distances between optimised points and between reference
   (original) points. Use `target.valid_map` to determine which comparisons are
   valid. If `valid_map` is `True`, filter to only point pairs separated by
   `target.square_size`.
2. Estimate a **uniform scale factor** `s = mean(ref_distances /
   new_distances)`.
3. Scale the optimised points by `s`.
4. Estimate a **rigid transform** (rotation + translation) from the scaled
   optimised points to the reference points using
   `n_estimate_rigid_transform()` (SVD-based Procrustes).
5. Apply the rigid transform to the points, poses, and extrinsics.
6. If the rigid transform estimation fails, fall back to the identity.

**Data in → Data out (whole Phase 4):** Phase 3 `CameraSet` +
`TargetDetection` + `AbstractTarget` → refined `CameraSet` with jointly
optimised target geometry.

**Artefact:** User-defined save (e.g. `self_calib_test.camset`).

**Phase 4 diagnostics — see [Per-Phase Diagnostic Proposals](#per-phase-diagnostic-proposals).**

---

## Phase 5 — Visualisation & Validation

**Source:**
[`utils/visualisation.py` L170–334](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/utils/visualisation.py#L170-L334),
[`standard_bundle_handler.py` `special_plots()` L414–480](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/standard_bundle_handler.py#L414-L480)

**What happens:**

1. **Reprojection error scatter** — plot residuals as a 2D cluster (x-error vs
   y-error) with 1σ/2σ/3σ covariance ellipses.
2. **Per-camera coverage and error** — for each camera, detected point locations
   coloured by polarised Euclidean reprojection error (radial bias shown via
   coolwarm colourmap: red = error away from principal point, blue = towards).
3. **3D reconstruction** — triangulate detected points from multiple views.
4. **Outlier rejection in 3D** — reject points outside 3× the mean target
   distance from origin in target space.
5. **Point cloud visualisation** — PyVista 3-panel plot:
   - Panel 0: reconstructed points in scene coordinates with camera wireframes.
   - Panel 1: reconstructed points in target coordinates, coloured by
     reprojection error.
   - Panel 2: accuracy vs precision scatter for each unique feature (mm).
6. **Self-calibration specific (`special_plots`)** — if Phase 4 was run, show
   arrows on the target model indicating the recovered shape change, with
   per-face planarity RMS.

---

## Resolved Uncertainties

### Uncertainty A — Is `outlier_rejection()` in `camera_calibrator.py` dead code?

**Resolved: Yes, it is dead code in the current pipeline.**

The function `outlier_rejection()` at
[`camera_calibrator.py` L173–207](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/calibration/camera_calibrator.py#L173-L207)
is defined but **never called**. Its only potential call site, inside
`run_stereo_calibration()` at line ~254, is commented out:

```python
# outlier_rejection(optimisation.fun.reshape((-1,2)), param_handler)
```

This function operates on **post-optimisation residuals** — it groups residuals
by image number, computes per-image mean error, runs MAD at threshold 5, and
returns a pruned `TargetDetection`. It was likely intended for an
optimise-reject-reoptimise loop, but that loop was never implemented.

The **active** outlier path is `find_and_exclude_transform_outliers()` on
`TemplateBundleHandler`, which operates **pre-optimisation** during initial
parameter estimation (sub-phase 3b). It uses MAD at a higher threshold of 20.

**Summary of all outlier rejection mechanisms in the codebase:**

| Mechanism | Location | When | What it checks | Threshold | Status |
|-----------|----------|------|---------------|-----------|--------|
| `find_and_exclude_transform_outliers()` | `template_handler.py` | Pre-optimisation (3b) | Per-image reprojection error | MAD × 20 | **Active** |
| `outlier_rejection()` | `camera_calibrator.py` | Post-optimisation | Per-image mean residual | MAD × 5 | **Dead code** |
| `pose_in_detections()` outlier loop | `abstract_target.py` | On explicit call | Pose translation magnitude | MAD × 5 | **Not in standard pipeline** |
| `reject_outliers()` | `visualisation.py` | During visualisation | Per-point scatter | 2 × median deviation | **Visualisation only** |

### Uncertainty B — How does the abstract function block code generation system work?

**Resolved: It is a string-based code generation system that writes
numba-compiled Python files to disk.**

The system is implemented in `abstract_function_blocks.py` class
`optimisation_function`. Key mechanisms:

1. **Function block chain** — each optimisation is defined as a sequence of
   `abstract_function_block` subclasses, composed via `+`. For example:
   `projection() + extrinsic3D() + template_points()`.

2. **Loss function generation** (`make_full_loss_template`):
   - Constructs a filename from block names:
     `loss_projection_extrinsic3D_template_points.py`.
   - If the file exists in `template_functions/`, imports it directly.
   - If not, extracts the source of each block's `compute_fun` using
     `inspect.getsource()`, strips the function signature, and splices the body
     into a generated numba `@njit(parallel=True, fastmath=True, cache=True)`
     loop.
   - Writes to disk, then imports and returns a compiled callable.

3. **Jacobian generation** (`make_full_jac_template`):
   - Same pattern, but also generates a "matflow" function (chain-rule matrix
     multiplication) via `matmul_map.py`.
   - Handles fixed/unfixed parameter masking by mapping column indices and
     filtering the CSR sparse output.

4. **Caching** — generated `.py` files persist on disk. They are only
   regenerated if deleted or if `overwrite_function=True`.

---

## Remaining Open Questions

1. **`check_for_target_misalignment()`** — at
   [`template_handler.py` L428–452](https://github.com/ColDSnit/pyCamSet/blob/b180327e77a2679dcb371bbebea2ee2792f16557/pyCamSet/optimisation/template_handler.py#L428-L452).
   Checks for inconsistent relative translations (stdev > 50 mm) and rotations
   (stdev > 5°). **Commented out** in `estimate_camera_relative_poses()`.

2. **`cameras_converged` loop** — at ~L523 of `template_handler.py`. A
   commented-out iterative camera-pose-refinement scheme.

3. **`cam_good` check** — at ~L567. A commented-out per-camera quality gate.

4. **`reasonable_bound` guard** — at ~L557. A commented-out guard against
   implausibly high costs per image.

5. **Solver choice** — `method="lm"` and `loss="cauchy"` are commented out in
   `run_bundle_adjustment()`. The active solver is TRF.

---

## Per-Phase Diagnostic Proposals

For each phase, the following diagnostics are proposed for the
`phased_calibration` module. Each diagnostic is classified as:

- **Q** = Quantitative (produces a number or pass/fail)
- **V** = Visual (produces a plot or image)

### Phase 0 diagnostics

| ID | Type | Diagnostic | Implementation notes |
|----|------|-----------|---------------------|
| D0.1 | Q | Number of camera subfolders found | `len(get_subfolder_names(f_loc))` |
| D0.2 | Q | Number of images per camera | Count files via `glob_ims()` per subfolder |
| D0.3 | Q | Image count consistency | Boolean: all cameras have equal image count |

### Phase 1 diagnostics

| ID | Type | Diagnostic | Implementation notes |
|----|------|-----------|---------------------|
| D1.1 | Q | Total detections per camera | `TargetDetection.get_cam_list()` → count rows per camera |
| D1.2 | Q | Detection rate per camera (%) | Already computed by `validate_detections()`: fraction of images where board was found |
| D1.3 | Q | Board completeness per camera (%) | Already computed by `validate_detections()`: mean fraction of expected corners found per detection |
| D1.4 | V | Features-per-image-per-camera heatmap | `features_per_im_per_cam()` returns a (n_images × n_cameras) matrix. Display as `imshow`. Rows = images, columns = cameras. Zero entries (missed detections) are immediately visible. |
| D1.5 | V | Per-camera detection overlay montage | Composite of detected corners overlaid on a sample image per camera. Already partially supported by `draw=True`. |
| D1.6 | Q | Per-camera detection spatial coverage | Compute convex hull area of detected 2D points divided by image area. Low coverage suggests the target was only seen in a small region of the sensor, which weakens the intrinsic estimate. |
| D1.7 | Q | Minimum features in any single image-camera pair | `features_per_im_per_cam().min()`. If below ~6, that observation may be unreliable for PnP. |

### Phase 2 diagnostics

| ID | Type | Diagnostic | Implementation notes |
|----|------|-----------|---------------------|
| D2.1 | Q | Per-camera RMS reprojection error from `cv2.calibrateCamera` | `cv2.calibrateCamera()` returns the overall RMS. pyCamSet does not currently capture this value. The `calibrateCameraExtended` variant also returns `perViewErrors` and `stdDeviationsIntrinsics` — these could be captured if the call were modified. |
| D2.2 | Q | Per-camera focal length (fx, fy) and principal point (cx, cy) | Read directly from the resulting `Camera.intrinsic` matrix. Display as a table. Flag cameras whose principal point is far from image centre or whose focal lengths differ significantly from the population mean. |
| D2.3 | Q | Per-camera distortion coefficient magnitudes | `Camera.distortion_coefs`. Large higher-order coefficients may indicate overfitting if the board coverage was poor. |
| D2.4 | V | Per-camera undistorted grid | Apply `cv2.undistort` to a synthetic grid image using the estimated intrinsics. Visual inspection reveals barrel/pincushion distortion and whether the model is physically plausible. |
| D2.5 | Q | Intrinsic parameter standard deviations | If `cv2.calibrateCameraExtended` is used, OpenCV returns `stdDeviationsIntrinsics` — estimated standard deviations for `[fx, fy, cx, cy, k1, k2, p1, p2, k3, ...]` derived from the Jacobian covariance. Large standard deviations relative to the parameter value indicate poor constraint. pyCamSet does not currently request these outputs. |
| D2.6 | Q | Per-view reprojection error from OpenCV | Similarly, `perViewErrors` from `calibrateCameraExtended` gives per-image RMS. Outlier images can be identified here before proceeding to Phase 3. |
| D2.7 | V | Per-camera reprojection error vs image index | Plot per-view RMS error as a line/bar chart. Spikes indicate problematic images. |

### Phase 3 diagnostics

| ID | Type | Diagnostic | Implementation notes |
|----|------|-----------|---------------------|
| D3.1 | Q | Number of images excluded as missing poses | `np.sum(missing_poses)` |
| D3.2 | Q | Number of images excluded by outlier rejection | Track before/after count in `find_and_exclude_transform_outliers()` |
| D3.3 | Q | Per-image initial reprojection error (pre-optimisation) | `per_im_error` array from `estimate_camera_relative_poses()`. Already computed but not persisted. |
| D3.4 | V | Per-image initial error bar chart | Visualise D3.3 as a bar chart, with outlier threshold line overlaid. |
| D3.5 | Q | Initial Euclidean reprojection error (px) | Already logged in `run_bundle_adjustment()`: `init_euclid`. |
| D3.6 | Q | Final Euclidean reprojection error (px) | Already logged: `final_euclid`. |
| D3.7 | Q | Error reduction ratio | `init_euclid / final_euclid`. Values close to 1.0 suggest the optimiser made little progress (possible convergence issue or already-good initialisation). |
| D3.8 | Q | Solver termination status | `optimisation.status` and `optimisation.message` from `scipy.optimize.least_squares`. Status 0 = max iterations reached (possibly unconverged). Status 1–3 = various convergence criteria met. |
| D3.9 | Q | Number of function evaluations used | `optimisation.nfev`. Compare to `max_nfev`. If equal, the solver hit the limit. |
| D3.10 | Q | Parameter count vs observation count | `len(init_params)` vs `len(init_err) // 2`. The ratio should be well below 1.0 for the problem to be well-constrained. |
| D3.11 | V | Residual x vs y scatter | Already implemented in `visualise_calibration()` via `cluster_plot()`. Systematic elliptical or biased structure indicates unmodelled distortion or misalignment. |
| D3.12 | Q | Per-camera mean reprojection error | Decompose the final residual vector by camera index. Flag cameras with error significantly above the population median. |
| D3.13 | V | Extrinsic pose visualisation | Plot camera wireframes and target poses in 3D. Already partially supported by `CameraSet.get_scene()`. Useful for sanity-checking that cameras are in physically plausible locations. |

### Phase 4 diagnostics

| ID | Type | Diagnostic | Implementation notes |
|----|------|-----------|---------------------|
| D4.1 | Q | Number of free (unfixed) target points | `np.sum(visible_feature_mask)` — points actually being optimised. |
| D4.2 | Q | Number of gauge-fixed points | Always 3 (7 DOF fixed). Report which point indices were chosen. |
| D4.3 | Q | Initial and final Euclidean reprojection error | Same as D3.5/D3.6, but for the Phase 4 optimisation. |
| D4.4 | Q | Error reduction relative to Phase 3 | Compare Phase 4 `final_euclid` with Phase 3 `final_euclid`. The improvement is often modest (sub-pixel); if it worsens, the free-point optimisation may be overfitting. |
| D4.5 | Q | Gauge transform scale factor | The `s` computed in `apply_gauge_transform()`. Should be close to 1.0. Significant deviation suggests the optimiser drifted in scale. |
| D4.6 | Q | Gauge transform rotation magnitude | `np.linalg.norm(rodrigues(R))` of the rigid transform. Should be small (< few degrees). |
| D4.7 | Q | Mean Euclidean displacement of target points from nominal (mm) | `mean(||optimised_point - original_point||)` after gauge correction. Already computed inside `special_plots()`. |
| D4.8 | V | Target shape change arrows | Already implemented in `special_plots()` — arrows from original to optimised positions. |
| D4.9 | Q | Per-face planarity RMS (mm) | Already computed in `special_plots()` via `rms_plane()`. For a Ccube target, each face should be planar. Large RMS indicates the recovered geometry is not physically consistent. |
| D4.10 | V | Accuracy vs precision per feature | Already implemented in `visualise_calibration()` Panel 2 — scatter of mean distance from expected position vs mean scatter about the mean position. Points above the y=x line are more precise than accurate (possible systematic bias). |

### Phase 5 diagnostics

Phase 5 is itself the diagnostic layer. The proposals above aim to bring
Phase-5-quality analysis to the end of **every** phase, not just the final one.

---

## Summary Table

| Phase | Sub-phase | Function(s) | What It Does | Key Artefact |
|-------|-----------|-------------|--------------|--------------|
| 0 | — | `calibrate_cameras()` top | Validate inputs | Clean parameters |
| 1 | 1a | `get_subfolder_names()`, `sanitise_input_images()` | Discover cameras, check image counts | — |
| 1 | 1b | `detect_datapoints_in_imfile()` → `find_in_image()` | Detect target corners | `detected_datapoints.pickle` |
| 1 | 1c | `validate_detections()` | Quality check | Log warnings |
| 2 | 2a | `features_per_im_per_cam()` | Select best-pose image | — |
| 2 | 2b | `run_initial_calibration()` → `cv2.calibrateCamera()` | Per-camera intrinsics | `initial_cameras.camset` |
| 2 | 2c | (conditional) re-detect with undistortion | High-distortion refinement | `initial_cameras_high_distortion.camset` |
| 3 | 3a | `TemplateBundleHandler.__init__()` | Set up optimisation structure | Handler object |
| 3 | 3b | `calc_initial_params()` → `estimate_camera_relative_poses()` | Pose estimation + outlier rejection | Initial parameter vector |
| 3 | 3c | `make_optimisation_function()` | Compile loss + Jacobian | Numba-compiled callables |
| 3 | 3d | `run_bundle_adjustment()` → `scipy.optimize.least_squares` | Joint optimisation (fixed target) | `optimised_cameras.camset` |
| 4 | 4a | `SelfBundleHandler.__init__()` | Extend handler with free points | Handler object |
| 4 | 4b | `set_from_templated_camset()` | Warm-start from Phase 3 | Extended parameter vector |
| 4 | 4c | `run_bundle_adjustment()` | Joint optimisation (free target) | Final `CameraSet` |
| 4 | 4d | `apply_gauge_transform()` | Scale + rigid alignment to reference | Gauge-corrected output |
| 5 | — | `visualise_calibration()` + `special_plots()` | Analyse + display results | Plots + 3D scatter |

---

## Source File Index

| File | Key contents |
|------|-------------|
| `pyCamSet/calibration/camera_calibrator.py` | `calibrate_cameras()`, `detect_datapoints_in_imfile()`, `run_initial_calibration()`, `validate_detections()`, `run_stereo_calibration()`, `outlier_rejection()` (dead) |
| `pyCamSet/calibration_targets/abstract_target.py` | `AbstractTarget`, `find_in_imfolder()`, `find_in_image()`, `initial_calibration()`, `target_pose_in_cam_image()`, `pose_in_detections()` |
| `pyCamSet/calibration_targets/target_Ccube.py` | `Ccube` target implementation, ArUco/ChArUco detection |
| `pyCamSet/calibration_targets/target_detections.py` | `ImageDetection`, `TargetDetection` data containers |
| `pyCamSet/optimisation/template_handler.py` | `TemplateBundlePrimitive`, `TemplateBundleHandler`, `estimate_camera_relative_poses()`, `check_feasiblity_and_update_refpose()`, `check_for_target_misalignment()` |
| `pyCamSet/optimisation/standard_bundle_handler.py` | `StandardBundlePrimitive`, `SelfBundleHandler`, `apply_gauge_transform()`, `special_plots()` |
| `pyCamSet/optimisation/optimisation_handling.py` | `make_optimisation_function()`, `run_bundle_adjustment()` |
| `pyCamSet/optimisation/abstract_function_blocks.py` | `optimisation_function`, `abstract_function_block`, code generation for loss/Jacobian |
| `pyCamSet/optimisation/function_block_implementations.py` | `projection()`, `extrinsic3D()`, `template_points()`, `rigidTform3d()`, `free_point()` |
| `pyCamSet/optimisation/matmul_map.py` | Chain-rule matrix flow code generation for Jacobian |
| `pyCamSet/optimisation/compiled_helpers.py` | Numba-compiled helpers: transforms, cost function evaluation |
| `pyCamSet/utils/general_utils.py` | `mad_outlier_detection()`, transform utilities |
| `pyCamSet/utils/visualisation.py` | `visualise_calibration()`, error plots, 3D reconstruction visualisation |
| `pyCamSet/cameras/camera.py` | `Camera` object (intrinsics, extrinsics, distortion, projection) |
| `pyCamSet/cameras/camera_set.py` | `CameraSet` container, triangulation, serialisation |
| `pyCamSet/utils/saving.py` | `save_pickle()`, `load_pickle()`, `load_CameraSet()` |