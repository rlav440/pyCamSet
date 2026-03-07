# Merge Investigation: calibria/pcss → pyCamSet

**Date:** 2026-03-07  
**Branch:** `copilot/merge-calibria-into-pycamset`  
**Scope:** Map every calibria/pcss dependency on pyCamSet, identify minimal edits needed,
propose a new `pyCamSet/pipeline/` module layout, and list what stays in calibria.

---

## 1. Touchpoint Map

Every calibria/pcss file and the exact pyCamSet symbols it touches.

| calibria/pcss file | pyCamSet symbol | pyCamSet module |
|---|---|---|
| `pipeline.py` | `TargetDetection`, `ImageDetection` | `pyCamSet.calibration_targets` |
| `pipeline.py` | `SelfBundleHandler` | `pyCamSet.optimisation.standard_bundle_handler` |
| `pipeline.py` | `run_bundle_adjustment` | `pyCamSet.optimisation.optimisation_handling` |
| `pipeline.py` | `detect_datapoints_in_imfile` | `pyCamSet.calibration.camera_calibrator` |
| `pipeline.py` | `run_initial_calibration` | `pyCamSet.calibration.camera_calibrator` |
| `pipeline.py` | `run_stereo_calibration` | `pyCamSet.calibration.camera_calibrator` |
| `pipeline.py` | `mad_outlier_detection` | `pyCamSet.utils.general_utils` |
| `io_helpers.py` | `load_CameraSet` | `pyCamSet.utils.saving` |
| `io_helpers.py` | `CameraSet.save()` | `pyCamSet.cameras.camera_set` (via instance) |
| `plot_helpers.py` | `CameraSet.get_scene()` | `pyCamSet.cameras.camera_set` (via instance) |
| `plot_helpers.py` | Edited copy of `cluster_plot` | `pyCamSet.utils.visualisation` |
| `plot_helpers.py` | Edited copy of `fancy_confidence_contours` | `pyCamSet.utils.visualisation` |
| `plot_helpers.py` | Edited concept of `visualise_calibration` | `pyCamSet.utils.visualisation` |
| `gui/state.py` | *(none — pure GUI state dataclass)* | — |
| `gui/workers.py` | All six phase functions via `calibria.pcss.pipeline` | `calibria.pcss.pipeline` (indirect) |

### Additional calibria-internal dependencies in `pipeline.py`

| calibria symbol | Role |
|---|---|
| `calibria.detection.load_and_normalize_image` | Raw image loader (raises on bad file) |
| `calibria.detection.get_images_in_folder` | Sorted image path lister |
| `calibria.pcss.io_helpers.*` | JSON / CSV / pickle / camset file I/O |
| `calibria.pcss.plot_helpers.*` | Error histogram and coverage plots |

---

## 2. Edited-Copy Diffs

Three functions in `pcss/pipeline.py` are reimplementations of pyCamSet originals.
Each diff below lists what changed and why.

---

### 2a. `_sanitise_input_images` (pcss) vs `sanitise_input_images` (pyCamSet)

**pyCamSet original** — `pyCamSet/calibration/camera_calibrator.py`, line 401:

```python
def sanitise_input_images(detected_sub_folders: list[Path], optmode: str = 'na'):
    equal_ims = [len(glob_ims(fol)) for fol in detected_sub_folders]
    if not len(set(equal_ims)) <= 1:
        raise ValueError("An unequal number of calibration images were passed in the input folders.")
```

**pcss edited copy** — `pcss/pipeline.py`:

```python
def _sanitise_input_images(cam_folders: List[Path]) -> None:
    counts = {f.name: len(_images_in_folder(f)) for f in cam_folders}
    if len(set(counts.values())) > 1:
        raise ValueError(f"Unequal image counts across camera folders: {counts}")
```

| What changed | Why |
|---|---|
| `glob_ims()` replaced by `_images_in_folder()` | calibria uses its own folder-lister; avoids importing pyCamSet's glob utility directly |
| Error message includes the actual counts | More informative for debugging |
| `optmode` parameter removed | pcss never used optional mode |
| Returns `None` explicitly | Style consistency |

---

### 2b. `_validate_detections` (pcss) vs `validate_detections` (pyCamSet)

**pyCamSet original** — `pyCamSet/calibration/camera_calibrator.py`, lines 354–398:

```python
def validate_detections(detected: TargetDetection, target: AbstractTarget):
    # ... iterates, logs per camera, returns None
    return
```

**pcss edited copy** — `pcss/pipeline.py`:

```python
def _validate_detections(detected: TargetDetection, target) -> Dict[str, Dict]:
    # ... same iteration, but returns {cam: {detection_rate, mean_board_completeness}}
    return stats
```

| What changed | Why |
|---|---|
| Returns `Dict[str, Dict]` instead of `None` | pcss caches validation stats to JSON; `None` return is not serialisable |
| `logging.warning` calls preserved | Same detection-quality warnings emitted |
| `board_fraction` dict uses `setdefault` | Minor style difference; same logic |

---

### 2c. `_outlier_rejection_auto` (pcss) vs `outlier_rejection` (pyCamSet)

**pyCamSet original** — `pyCamSet/calibration/camera_calibrator.py`, lines 192–227:

```python
def outlier_rejection(results, params) -> tuple[TargetDetection | None, bool]:
    detection = params.get_detection_data()
    d_list = [[] for _ in range(params.detection.max_ims)]
    for im_num, errs in zip(detection[:, 1], results):
        d_list[int(im_num)].append(errs)
    per_im_outliers = mad_outlier_detection([np.mean(datum) for datum in d_list if datum],
                                            draw=False, out_thresh=5)
    if per_im_outliers is not None:
        plt.boxplot(d_list)
        plt.show()   # <— interactive blocking call
    # ...
    return data.delete_row(im_num=per_im_outliers), True
```

**pcss edited copy** — `pcss/pipeline.py`:

```python
def _outlier_rejection_auto(per_image_means: List[float],
                             out_thresh: float = 5.0) -> Optional[List[int]]:
    from pyCamSet.utils.general_utils import mad_outlier_detection
    result = mad_outlier_detection(per_image_means, out_thresh=out_thresh, draw=False)
    if result is None:
        return None
    return [int(i) for i in result[0]]
```

| What changed | Why |
|---|---|
| `plt.show()` removed | pcss must be headless (GUI and CLI) |
| `input()` prompt removed | Non-interactive pipeline; caller decides what to do with flagged indices |
| Accepts pre-computed `per_image_means` list | Caller computes means in Phase 4; avoids tight coupling to `params` object |
| Returns `List[int]` instead of `(TargetDetection, bool)` | Phase 4 just needs flagged indices; pruning is done separately |

---

### 2d. `plot_residual_clusters` / `_fancy_confidence_contours` (pcss) vs pyCamSet originals

**pyCamSet originals** — `pyCamSet/utils/visualisation.py`, lines 17–102 and 104–166:

```python
def cluster_plot(data_list, ranges=None, titles=None, alphas=None, s_per=None, save=None):
    # requires pyvista and scipy.stats at module level
    # calls plt.show()
    ...

def fancy_confidence_contours(x, y, ax, ranges):
    ...
```

**pcss edited copies** — `pcss/plot_helpers.py`:

```python
def plot_residual_clusters(data_list, titles=None, out_dir=None, title='Residual Cluster'):
    # deferred matplotlib import; no pyvista or scipy.stats
    # calls save_figure(); does NOT call plt.show()
    ...

def _fancy_confidence_contours(x, y, ax, ranges):
    # same maths; pyvista import removed
    ...
```

| What changed | Why |
|---|---|
| `plt.show()` removed | pcss plots must be saveable without blocking |
| `save=None` replaced by `out_dir + save_figure()` | Consistent with pcss save convention |
| pyvista / scipy.stats module-level imports removed | Optional dependencies; deferred inside function |
| Added `title` parameter | Used to derive the saved `.png` filename via `title_to_filename()` |

---

### 2e. `plot_coverage_scatter` (pcss) vs `visualise_calibration` (pyCamSet)

**pyCamSet original** — `pyCamSet/utils/visualisation.py`, lines 170–334:  
`visualise_calibration` is a monolithic function that recomputes errors from `param_handler` internals and renders multiple interactive pyvista panels.

**pcss reimplementation** — `pcss/plot_helpers.py`:  
`plot_coverage_scatter` accepts pre-computed `(u, v, error)` tuples from the Phase 4 CSV cache, draws a single camera's coverage scatter, and optionally saves it.

| What changed | Why |
|---|---|
| Accepts CSV data instead of `param_handler` | Works from cached Phase 4 output; no live handler needed |
| Single-camera function | GUI can call per-camera to update individual panels |
| pyvista panels removed | Replaced by matplotlib scatter (portable, headless) |

---

## 3. Proposed pyCamSet Edits

Three specific, minimal changes to pyCamSet core.

---

### 3a. `run_stereo_calibration` — set resolutions whenever `floc` is provided

**File:** `pyCamSet/calibration/camera_calibrator.py`, lines 276–279

**Status: ✅ Applied in this PR.**

**Before:**

```python
if save:
    if floc is not None:
        optimised_cams.set_resolutions_from_file(floc)
    optimised_cams.save(save_loc)
```

**After (applied):**

```python
if floc is not None:
    optimised_cams.set_resolutions_from_file(floc)
if save:
    optimised_cams.save(save_loc)
```

**Rationale:** When `run_stereo_calibration` is called externally (e.g. from the new phased pipeline or from user scripts) with `save=False` and `floc` supplied, the camera resolutions were never populated. `calibrate_cameras()` worked around this by calling `initial_cams.set_resolutions_from_file()` before `run_stereo_calibration`, but that pattern was not enforced when the function was used standalone. The fix decouples resolution-setting from saving.

**Backwards-compatibility:** Fully backwards-compatible. When `floc=None` (the common default), behaviour is unchanged. When `floc` is supplied and `save=True`, behaviour is the same as before. When `floc` is supplied and `save=False`, resolutions are now correctly set (previously they were silently skipped).

---

### 3b. `problem_options` forwarding chain — already correct

**Verification:** Tracing the call chain:

```
calibrate_cameras(problem_options=...)
  └─ run_stereo_calibration(..., problem_options=problem_options)   # line 124
       └─ TemplateBundleHandler(..., options=problem_options)        # line 263
            └─ self.problem_opts = DEFAULT_OPTIONS                   # line 114
               self.problem_opts.update(options)                     # line 116
```

No silent `None` paths exist in the standard flow.  **No change needed.**

*Note:* `DEFAULT_OPTIONS` is a module-level mutable dict (line 29 of `template_handler.py`).  Calling `.update(options)` on it mutates the shared default for subsequent instantiations.  This is a pre-existing bug unrelated to this merge; it should be fixed separately by using `dict(DEFAULT_OPTIONS)` as the base copy.

---

### 3c. `calibration_handler` round-trip through `.camset` — works, with caveats

**Verification of `load_CameraSet`** (`pyCamSet/utils/saving.py`, lines 151–236):

The save/load cycle correctly reconstructs `calibration_handler`:

1. `save_camset` writes `handler_module`, `handler_name`, `fixed_params`, `options`, and `missing_poses` to JSON (lines 112–121).
2. `load_CameraSet` reads these back and calls `instance_obj(handler_module, handler_name, ...)` (lines 220–222).
3. The loaded handler is a fresh `TemplateBundleHandler` (or `SelfBundleHandler`) instance with the same args.
4. `SelfBundleHandler.set_from_templated_camset()` checks `isinstance(prev_cams.calibration_handler, TemplateBundleHandler)` (line 271); a reloaded `TemplateBundleHandler` passes this check.

**Caveat — detection data:** `load_CameraSet` decompresses the `TargetDetection` from the blosc-compressed `dtct_config.compressed_data` field.  If the detection class name or module changes between pyCamSet versions, the `instance_obj` call will fail gracefully (line 195–197, logging a warning and returning just the CameraSet).  **No change needed** for the merge, but the pipeline should always call `set_from_templated_camset()` immediately after loading, before any detection-data operations.

---

## 4. Proposed New File Layout

All pipeline logic moves to a new `pyCamSet/pipeline/` sub-package.  Plot helpers
live in the existing `pyCamSet/utils/visualisation.py` so that they are available
to all pyCamSet users, not only pipeline callers.

```
pyCamSet/
├── utils/
│   └── visualisation.py    — Interactive tools (unchanged) + headless pipeline plot helpers
└── pipeline/
    ├── __init__.py          — Public API re-exports (phases, cache helpers, plot helpers)
    ├── phased_pipeline.py   — Phases 1–6 + run_pipeline() orchestrator
    └── pipeline_cache.py    — JSON / CSV / pickle / camset caching helpers
```

### Why put the plot helpers in `visualisation.py` rather than a separate file?

The new headless helpers (`save_figure`, `plot_error_histogram`, `plot_per_camera_errors`,
`plot_residual_clusters`, `plot_coverage_scatter`, `plot_camera_arrangement`) are general
enough to be useful outside the pipeline context — for example, in notebooks or custom
scripts that call `run_stereo_calibration` directly.  Keeping them in `visualisation.py`
makes them discoverable alongside the existing `cluster_plot` and `visualise_calibration`
functions.  `pipeline/__init__.py` re-exports them for convenience.

### Should `calibrate_cameras()` become a thin wrapper?

**Recommendation:** Keep `calibrate_cameras()` independent for now.  It is a well-tested, well-used entry point that follows a simpler (non-cached) code path.  `run_pipeline()` in the new package can call the same underlying primitives (`run_initial_calibration`, `run_stereo_calibration`) directly, just as `calibrate_cameras()` does.  A future refactor could make `calibrate_cameras()` a thin wrapper over `run_pipeline()`, but that is out of scope for this merge.

---

## 5. Migration Checklist

Ordered steps to execute the merge safely.

- [x] **Step 0 — Read this document** and the calibria source in full.
- [x] **Step 1 — Create `pyCamSet/pipeline/`** sub-package with skeleton files (done in this PR).
- [ ] **Step 2 — Implement `pipeline_cache.py`** — copy `io_helpers.py` content, replace `calibria.*` imports with pure-stdlib / pyCamSet imports.  Run existing `calibration_test.py` to confirm nothing broke.
- [x] **Step 3 — Implement plot helpers in `pyCamSet/utils/visualisation.py`** — headless versions of `cluster_plot` / `visualise_calibration` and new helpers (`plot_error_histogram`, `plot_per_camera_errors`, `plot_coverage_scatter`, `save_figure`).  Done in this PR.
- [x] **Step 4 — Apply `run_stereo_calibration` fix** (3a above): decouple resolution-setting from `save` flag.  Done in this PR.
- [ ] **Step 5 — Implement `phased_pipeline.py`** phases 1–3 — translate from pcss, replace `calibria.*` image-loading helpers with pyCamSet equivalents (`detect_datapoints_in_imfile`, `glob_ims`).
- [ ] **Step 6 — Implement phases 4–6** — translate error analysis and self-calibration phases; use `pyCamSet.utils.general_utils.mad_outlier_detection` directly.
- [ ] **Step 7 — Implement `run_pipeline()`** orchestrator — wire phases 1–6 with consistent `out_dir` / `save_cache` / `load_cache` defaults.
- [ ] **Step 8 — Update `pyCamSet/pipeline/__init__.py`** to export the public API.
- [ ] **Step 9 — Write tests** for each phase in `tests/test_pipeline.py` using existing `test_data/` fixtures.
- [ ] **Step 10 — Update `calibria/pcss/pipeline.py`** to import from `pyCamSet.pipeline` instead of reimplementing.  Verify calibria tests still pass.
- [ ] **Step 11 — Update documentation** — add `pipeline` to the pyCamSet README and docsite.

### What to test at each step

| Step | Test |
|---|---|
| 2 | `save_json` / `load_json` / `save_pickle` / `load_pickle` round-trips |
| 3 | `plot_error_histogram` / `plot_per_camera_errors` return a Figure without calling `plt.show()` |
| 4 | `run_stereo_calibration(save=False, floc=<path>)` sets `cam.res` |
| 5–6 | Each phase returns the documented dict keys; no exceptions on test data |
| 7 | `run_pipeline()` end-to-end on `tests/test_data/` |
| 10 | All existing calibria GUI and headless tests pass without modification |

---

## 6. Backwards-Compatibility Notes

| Change | Impact | Mitigation |
|---|---|---|
| `run_stereo_calibration` now calls `set_resolutions_from_file` when `floc is not None` even if `save=False` | Previously silently skipped.  Only affects users who pass `floc` but `save=False` — this combination was previously broken (resolutions not set). | **Applied in this PR.** New behaviour is correct.  No default changes. |
| New `pyCamSet/pipeline/` sub-package added | No existing imports break; purely additive. | — |
| `validate_detections` returns `None` in pyCamSet; pcss wrapper returns `Dict` | No conflict — they are separate functions. | Keep both; pcss wrapper delegates to pyCamSet log messages and adds JSON output. |
| `.camset` save format unchanged | Merge does not touch `save_camset` or `load_CameraSet`. | — |
| `calibrate_cameras()` signature unchanged | Merge does not modify the existing entry point. | All new parameters in new pipeline functions have defaults preserving current behaviour. |
| `DEFAULT_OPTIONS` mutability bug | Pre-existing issue in `template_handler.py`; not introduced by this merge. | Fix separately: change line 114 to `self.problem_opts = dict(DEFAULT_OPTIONS)`. |

---

*End of investigation. Skeleton files are created under `pyCamSet/pipeline/`.*
