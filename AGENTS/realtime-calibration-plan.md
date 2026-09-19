# Real-time (streaming) calibration in pyCamSet — implementation plan v0.3

Status: **proposal for review. No code written.** v0.3 supersedes v0.2, which
carried an unsupported "~2 fps" sizing premise, a thread count derived from it, and
an index-correspondence guard that was necessary but not sufficient (all three
corrected in §0.1 and §0.3.2). v0.2 in turn superseded v0.1, which contained one
wrong claim about the producer and a sizing premise taken from the wrong acquisition
regime (both retracted in §0.1 and §0.2).

Target rig for first deployment: **M_NEBULA** (8 x Basler acA720-520um, hardware
triggered, 4 x DLP4500). Framework: pyCamSet fork `ColDSnit/pyCamSet`, branch
`development`, checkout `b79bb76`.

Provenance tags: `[VERIFIED-BY-READING]` (source read this session),
`[VERIFIED-BY-EXECUTION]` (computed, result quoted), `[INFERENCE]` (reasoned from
verified facts), `[ASSUMED — owner]` (owner-stated, not verified in code),
`[OPEN]` (not established).

### v0.3 changelog — what changed after S0 was actually run

S0 is no longer a question (§7); it was run on the rig's own 797-frame acquisition and
its findings are folded in below. Only four things moved, all of them away from an
assumption and toward a measurement:

| Was | Now | Where |
|---|---|---|
| sized at "~2 fps" | the one measured rate is **4.95 fps**; the plan no longer depends on a rate | §0.1 |
| Tier 1 pool "2-4 threads" | pool = **one worker per camera (8)**; the tick is a barrier, so the budget is the frame period | §0.1, §4 |
| index *i* simultaneous `[ASSUMED — owner]` | **measured**: identical start + contiguous run; short camera excluded, never offset-corrected; per-frame simultaneity needs BlockID persisted | §0.3.2, §8.2 |
| the rig's target unknown / assumed ChArUco2 | **Ccube2, 6 pts, 10 mm** on the rig's own artifacts; ChArUco2 was only a default | §0.2, §8.1 |

Everything else — the architecture, the store, the arrival rule, the tiers, the
metrics — stands as v0.2 wrote it, with two measured corrections inside it (arrival
rule §2: `metadata.json` trails the last frame by 6.0 s, so the end-of-run sweep needs
a stability backstop; §0.2: no per-frame time is persisted anywhere, so the stream
must timestamp arrivals itself).

Owner decisions recorded 2026-09-19 — see §8. In short: calibration acquisition at
720x540 Mono16, **measured 5 fps (the owner-stated ~2 fps is not supported by any
artifact — §0.1)**, 20 ms exposure; rig static; warm-start-from-a-prior-calibration
deferred. Index correspondence is **no longer an assumption**: the folder shows the
cameras are start-aligned and contiguous, and the only way to prove per-frame
simultaneity from disk is to persist BlockID (§0.3.2).

---

## 0. Sizing and the producer-side contract

### 0.1 Frame and rate maths `[VERIFIED-BY-EXECUTION]`

Calibration acquisition is **720x540 Mono16** — the full sensor frame, chosen to see
as much of the target as possible, not the 540x440 used for mouse-heart capture.
Mono16 is what the control software writes
(`converter.OutputPixelFormat = PixelType_Mono16`, `camera_service.py:1056`).

| Quantity | Value |
|---|---|
| per frame | 0.742 MiB |
| per 8-camera time step | 5.93 MiB |
| at 5 fps — the only measured rate: images/s, write rate | 40 images/s, 29.7 MiB/s |
| 100 frames (20 s) | 800 images, 0.58 GiB |
| 500 frames (1.7 min) | 4000 images, 2.90 GiB |
| 1000 frames (3.3 min) | 8000 images, 5.79 GiB |
| 2000 frames (6.7 min) | 16000 images, 11.59 GiB |

**Retracted from v0.1.** v0.1 built its sizing on 387 fps at 540x440 Mono12 — taken
from the rig's acquisition-calculator campaign — and concluded that Tier 1 detection
would need ~31 cores and a mandatory frame subsample. **Those figures describe the
mouse-heart experiment rate, not the calibration acquisition, and are withdrawn as
sizing inputs.** Detection is not the bottleneck and no subsampling policy is
required (the configurable hook stays in §4 for a slower machine or a debug mode).

**Retracted from v0.2 — the "~2 fps" premise, and the thread count derived from it.**
v0.2 sized at 2 fps (16 images/s, 11.9 MiB/s) and concluded "a small worker pool
(2-4 threads) suffices". **No artifact supports 2 fps.** The rig's own files say 5:
`metadata.json` `Frequency: 5`, the acquisition sequence's `target.frequency: 5`, and
`Experiment Info.txt` — start 21:40:25, end 21:40:50, so 20.0 s over 99 frame
intervals = 202.0 ms = **4.95 fps**. A 2 fps acquisition of 100 frames spans 49.5 s;
this folder's wall clock is 2.5x too short for it. `[VERIFIED-BY-READING —
pyCamSet_paper/ccube2/5_corners/experiment_001; no acquisition at 2 fps exists on
disk to check the owner's figure against]`

**The rate is not ours to choose, so the pool must not depend on it.** Frequency is
whatever the operator configured (`ui/acquisition_calculator.py:84` permits
1-10 000 Hz), so any baked-in fps constant goes stale — and the same argument kills a
*derived* constant such as "4 threads". **Size the Tier 1 pool at the camera count
(8): one thread per camera.** Index correspondence makes each tick a barrier — the 8
cameras deliver together — so the tick's budget is the frame period, ~200 ms at 5 fps,
and `ceil(8/8) = 1` detection wave per tick. A wave costs the *slowest camera*, not
the median. Measured per-frame end-to-end on the rig's own frames: median 53.1 ms,
p95 84.3 ms, **worst single frame 134.25 ms (cam4, median 67.11 ms, p95 98.81 ms)** —
so one wave sustains **~7.5 fps** (1000/134.25 ms) and every camera's *median* fits a
200 ms period by ~3x (the slowest, cam4's 67.11 ms, by 2.98x), its *p95* by 2.02x, and
its worst frame by 1.49x. A smaller pool makes a wave `ceil(8/N)` detections deep, so
one slow frame becomes the *sum* of two (268 ms at the measured max) and the tick
slips; the barrier turns detection jitter into dropped ticks rather than local
slowness. Sized this way, a 2 fps acquisition costs the same pool with 3.7x headroom
at the worst measured frame (7.4x at cam4's median), so the v0.2 premise is not just
unsupported, it was unnecessary. `[VERIFIED-BY-EXECUTION — computed from
`_s0_measurement_ccube2/detection_cost_report.json`, 797 frames, 8 cameras]`

**One caution on the numbers above: they are a floor, not a typical case.** Detection
cost tracks how much of the cube is in view — across the 8 cameras, median cost
correlates with median points found at **r = 0.93** (59-62 ms on the cameras finding
~54-57 points, 40-41 ms on those finding 34-38). The pooled 53.1 ms median is the cost
of a *partial* cube view: 78 of 797 frames found nothing at all, and the back cameras
saw roughly half the points the front ones did. A run where the cube is well framed
for every camera costs the cam4 end of the table, so **size against the per-camera
maxima (67.11 ms median / 98.81 ms p95 / 134.25 ms worst), not the pooled median.**
`[VERIFIED-BY-EXECUTION — Pearson r computed over the report's per-camera fields]`

**Still true from v0.1.** The durable artifact of a session is detections, not
images, and the pipeline stays read → detect → append → release. The earlier RAM
alarm was overstated — 5.79 GiB for a 1000-frame acquisition is affordable — so
retaining some images for later features is a live option rather than forbidden. It
just buys nothing for the live path, since the frames are on disk and re-readable.

### 0.2 Producer-side facts, from the downstream control software

Source: `coldsnit/control_software/software-panoramic-refactored`, `main` HEAD
`9d69c382`, plus the S0 measurement work now merged to `main` at `4e602f3d`.
**MSOT runs `main`.** Comparability is decided by
`measurement_module_sha256` in the report, not by a commit pair — see §10.1 for
why a commit SHA cannot be that check (the tool's former
`expected_for_comparability` constant and its mismatch warning have both been
removed). pyCamSet stays `b79bb76`.
`services/camera_service.py` / `services/experiment_service.py`
`[VERIFIED-BY-READING]`:

- Filename: `{camera_name}_{YYYYmmdd_HHMMSS}_{frame_index:06d}.tiff`
  (`_format_image_filename`, :61-73). The timestamp identifies the **first capture of
  that acquisition**; the index is that camera's own frame position within it.
- Layout: `save_path/<camera_name>/<filename>` (:1064) — the per-camera subfolders
  pyCamSet expects. `.tiff` is supported by pyCamSet
  (`_SUPPORTED_IMAGE_SUFFIXES`, `utils/general_utils.py:178`).
- Frames are rotated 90 degrees clockwise before being saved (`_rotate_frame`,
  :1126). Detection and any sensor-space heatmap are in **rotated** coordinates.
- Each grabbed frame is written with `cv2.imwrite` **straight to its final
  filename**: no temp-then-rename, no manifest, no completion marker
  (:1137-1145). An `imwrite` failure is detected and raised. The arrival rule (§2) is
  therefore mandatory, not defensive.
- Acquisition length is bounded (`for i in range(num_frames)`, :1071).
- **No per-frame time is persisted.** `GetTimeStamp()` — the exposure-start tick,
  1 ns/tick on this camera — is read only into a log line in the single-shot path
  (`camera_service.py:1737-1747`); `_grab_frames` (:1056-1265) never reads it, and no
  chunk mode is configured anywhere in `services/`, `models/`, `ui/`. `metadata.json`
  carries no per-frame timing either (fields at `_save_metadata` :797, :823-835).
  Consequence for the session: **the stream must timestamp arrivals itself** (its own
  monotonic clock at admission) and must not claim a frame time from the folder. On
  the rig's own folder the mtimes are useless for this — exFAT quantises them to 2 s,
  so a 5 Hz acquisition shows a median interframe gap of 0.0 ms and a p95 of 2000 ms.
  `[VERIFIED-BY-READING + the S0 report's own cadence section]`
- `metadata.json` is written into the experiment folder and carries the calibration
  target parameters plus frequency, exposure and num_frames
  (`_save_metadata`, :797; fields at :823-835). **It is written on the success path at
  the very end** — 6.0 s *after* the last frame on the measured acquisition — so it
  is an end-of-run signal in the arrival rule (§2) and nothing more.

**Correction to v0.1 — the calibration save path.** v0.1 stated that
`ExperimentService._calibration_save_path` is pyCamSet's stream folder.
**Withdrawn.** `configure_calibration` (service :1019, controller
`controllers/experiment_controller.py:450`) and `set_calibration_mode` (:247) still
exist and are reachable programmatically, but **no UI caller was found for either**,
so in operational use calibration frames land in the normal auto-generated experiment
save path. This matches the owner's statement that no calibration mode remains: the
plumbing is now dead weight, not the intended entry point.
`[VERIFIED-BY-READING — grep across ui/, controllers/, services/, worktrees excluded]`

Two consequences, both load-bearing:

1. **`metadata.json` is written when the acquisition finishes.** `_save_metadata` is
   called on the success path at :633, after the firmware tail wait — it does **not**
   exist while frames are arriving. Live detection therefore cannot depend on it. For
   the first implementation pass the operator **selects the target in pyCamSet**
   (`calibration_targets/core/target_registry.py`) and nothing reads `metadata.json`;
   metadata-driven target selection and the cross-check are deferred to the follow-up
   card (§9).
2. **The target parameters now exist on both sides and must agree.** The control
   software's `models/calibration_target.py` is explicitly PyCamSet-free and
   serialises to the metadata dict (`CalibrationTargetParams.to_dict()`). Supported
   targets and their parameters: ChArUco 5x7 squares / 30.0 mm square, **ChArUco2**
   (the same field set, ArUco 2 backend), Ccube 6 points / 30.0 mm length / 0.10
   border, **Ccube2** (same field set, ArUco 2 backend), PuzzleBoard 105x148 / 2.0 mm
   square, PuzzleBoardCube 20 squares per side / 10.0 mm square. **Detecting against
   the wrong spec yields plausible-looking garbage** — this is the configured-geometry
   failure the REBELS doctrine warns about. Use the existing
   `describe_target_mismatch` / `target_mismatch_message` (`workflow/params.py:154`)
   to *report* a mismatch rather than proceed.

   **Which target the rig actually flies — measured, per §7's S0 run.** Every witness
   in `pyCamSet_paper/ccube2/5_corners/experiment_001` agrees on **Ccube2, 6 points,
   10.0 mm, border 0.10, DICT_4X4_1000**: its `detected_datapoints_aruco2.identity.json`
   (`target_spec.type = "Ccube2"`, `n_points 6`, `length 10.0`), four of its five
   `.pycamset_workspace` phase1 runs, and its phase2 runs. **The 30 mm Ccube length and
   the 5x7/30 mm ChArUco2 board in `models/calibration_target.py` are defaults, not
   measurements**: no filename anywhere under `E:\M_Nebula` contains `charuco`, none
   contains `30mm`, and `2 Calibration\targets_to_print\2D\` holds only
   `ccube2_5points_10mm.svg`, `ccube2_6points_10mm.svg` and four ArUco 1 /
   puzzle-board files. So the one target question S0 leaves to the bench is **which
   Ccube2** — 5 points or 6, and its printed edge length — and the rig's own artifacts
   say 6 points at 10 mm. `[VERIFIED-BY-READING]`

   **Corollary for the §0.1 numbers:** detection cost tracks points found (r = 0.93
   across the 8 cameras at the median: 59-62 ms on the cameras finding ~54-57 points,
   40-41 ms on those finding 34-38), so 49 ms is the cost of a *cheap* cube view. Size
   against the per-camera maxima, not the pooled median.

### 0.3 What follows from 0.2

1. **Identity is derivable from the filename alone** —
   `(camera_name, acquisition_timestamp, frame_index)`. This resolves the v0 blocker
   that detections were keyed on a natsorted position (`abstract_target.py:434`,
   `general_utils.py:283`), with no new metadata needed.
2. **Index correspondence: necessary, not sufficient — and the guard is now measured,
   not assumed.** The rig's own folder settles the cheap half. Enumerating every tiff
   index in every camera of `pyCamSet_paper/ccube2/5_corners/experiment_001`: all
   eight cameras start at index 0 and seven run 0-99 contiguous; `cam3_BackRight_859T`
   holds 0-96 contiguous with **no interior gaps**, i.e. it is **tail-truncated, not
   offset**. `experiment_002` shows the same shape on the same camera (99 files,
   ending 98). The producer numbers frames by files actually written
   (`next_image_index += 1`, `camera_service.py:1223`) and writes that index into the
   filename, so a late-starting camera would gap at the *head*; cam3 starts at 0 and
   stops early, which is a camera that began with the others and died mid-run.
   `[VERIFIED-BY-READING — per-camera index enumeration]`

   So the v0.2 guard (**compare frame counts at the end and flag a mismatch**) is
   *necessary but not sufficient*, and the v0.2 wording that a short camera "would
   silently shift that camera's indices against the others" is wrong for this shape:
   a tail-truncated camera loses the *tail of the acquisition*, and every frame it did
   write keeps its true index. The correct, sharper guard, which is cheap and stays in
   Tier 1:

   - **Require an identical index start (0) and a contiguous run in every camera**;
     a hole anywhere other than the tail means the index is not a time key and the
     camera must be refused, not offset-corrected.
   - **A short camera is excluded from interleaving, never re-indexed** — fabricating
     an offset would invent simultaneity that the folder cannot demonstrate. This is
     the one place where doing nothing is the correct behaviour.
   - This also matches pyCamSet's own detector: a folder whose cameras disagree is
     refused outright by `phase1` (`"Camera folders must contain equal non-zero image
     counts."`, `pyCamSet/workflow/phase1.py:286-288`, the check that failed this
     acquisition's first run once already). The stream should *not* bypass that
     refusal silently — it should surface it as the stop verdict, which §0.3.4 says
     it bypasses for a different reason (the per-folder detection call, not a
     relaxed count rule).

   **What no artifact can settle, and must therefore be either recorded or
   declared:** from a finished folder, nothing on disk proves that index *i* is the
   same *instant* in two cameras. The physical identity is the camera's own frame
   counter over the shared trigger train, `GetBlockID()`; `services/simultaneity_analysis.py`
   pairs on exactly that (`_pair_by_block_id`, :166) because index alignment broke on
   real hardware — four cameras carried one extra leftover frame, so index *k* of one
   camera was index *k-1* of another and "the k-th frames were DIFFERENT exposures",
   producing a 20,537,408 ns apparent skew on a rig whose paired residual is **54 ns**
   (the 20,537,408 ns figure is mis-stated in that docstring — 20,537,408 **ms** is
   5.7 h, while 2.0537408e11 ns is 205 s, itself longer than the acquisition it
   describes; the *order* is what matters, and the point stands without the number).
   Nothing persists BlockID next to the frames: `GetTimeStamp()` is read only into a
   log line in the single-shot path (`camera_service.py:1737-1747`) and the bulk
   acquisition path never reads it (`_grab_frames`, :1056-1265, no chunk mode
   configured anywhere in the repo). **Tier 2 may interleave index-aligned frames
   only under a stated assumption; the durable fix is to persist `(block_id, timestamp)`
   per saved frame in the producer, which costs nothing at acquisition time and makes
   the pairing provable instead of assumed.** `[VERIFIED-BY-READING]`

   **Why a short camera must be excluded rather than accommodated — the rig already
   paid for it.** The stream's guard is not a precaution; it closes an observed hole.
   `pyCamSet/workflow/phase1.py:286-288` refuses a folder whose cameras disagree
   ("Camera folders must contain equal non-zero image counts."), and that refusal is
   **exactly what happened on this acquisition**: its first phase1 run
   (`20260917_215812_70200d`) selected all 8 cameras and recorded
   `error: 'Camera folders must contain equal non-zero image counts.'`, with no
   detection pickle written. Every later run selected **7** cameras — cam3 dropped —
   and succeeded at 100 images each. That is also why
   `detected_datapoints_aruco2.identity.json`'s `cam_names` lists seven cameras and not
   eight: the detection cache records what was actually detected, and cam3 was
   deliberately excluded. **So the folder's own history is the argument for the guard:
   the alternative to excluding a short camera is not a richer solve, it is a refused
   run — or, worse, an index-aligned solve that silently mixes the tail of one camera's
   sequence with the mid-run of another's.**
3. **The old detection cache must stay a MISS for stream folders.** Existing identity
   is `{target_spec, cam_names, n_lim}` plus a pickle-bytes SHA-256
   (`camera_calibrator.py:471`, `cache_matches` :558). A growing `n_lim` would
   invalidate the whole cache per frame — the opposite of the goal. The streaming
   store is a **new** artifact; `detected_datapoints*.pickle` keeps its current
   meaning and is never read as a partial hit. `[VERIFIED-BY-READING]`
4. **Phase 1's equal-non-zero-image-count requirement is bypassed, not changed**
   (`workflow/phase1.py:286`; `sanitise_input_images`, `camera_calibrator.py:1003`).
   The stream calls `detect_datapoints_in_imfile` per selected folder with an explicit
   `cam_names` list (`camera_calibrator.py:759`), which also avoids
   `staged_camera_root`'s whole-folder `copytree` fallback on Windows
   (`workflow/detections.py:195-204`). `[VERIFIED-BY-READING]`
5. **One acquisition per camera is in flight at a time, but write order is not
   guaranteed.** The camera loop is sequential — save inside the grab loop
   (`camera_service.py:1042`) — so the file *count* grows smoothly; however a frame
   that finished writing early can carry a higher index than one still flushing. The
   arrival rule must therefore test per-file stability, not "wait for the highest
   index". `[INFERENCE from the verified loop structure]`

---

## 1. Architecture

The fork already separates backend from GUI: `pyCamSet/workflow/` imports no UI
toolkit, and the seam is enforced by `tests/test_workflow_backend_seam.py`, whose
docstring states "nothing here may grow one".

New modules:

| Module | Responsibility |
|---|---|
| `pyCamSet/workflow/stream.py` | `StreamSession`: watcher, arrival rule, watermark, phase policy, cadence, failure isolation, checkpoint |
| `pyCamSet/workflow/stream_store.py` | Incremental detection store: append, load, prune, coverage/selection queries |
| `pyCamSet/gui/realtime_calibration_tab.py` | Thin subscriber: a `QTimer` polling a state snapshot |

Reused unchanged: `detect_datapoints_in_imfile`, `phase2.calibrate`
(`workflow/phase2.py:172`), `phase3.solve` (`workflow/phase3.py:241`),
`phase4.solve` (`workflow/phase4.py:105`), `workflow/diagnostics.py`.

Hard invariant: the tab owns no scheduling, no state and no solving. A
`StreamSession` must run from a script with no Qt importable; the GUI is a rendering
of it. That is what makes the feature testable, and it is the owner's stated
requirement.

### 1.1 Session lifecycle

```
StreamSession(folder, target_spec, policy)   # no I/O yet
  .start()      # watcher begins Tier 1 as images appear
  .snapshot()   # immutable state for a renderer: counts, metrics, flags
  .checkpoint() # materialise ordinary phase runs (§6)
  .stop()
```

`policy` holds the per-phase options the four dropdowns expose, plus camera
selection. It is **re-read every tick**, not latched at start: a change applies at the
next cadence boundary with no restart. This is deliberate — latched option lists are
what produced the duplicated-workflow problems in the existing phase GUIs.

Camera selection is **all present camera folders, minus a denylist**, with an
override list available. A one-time inclusion list is stale the moment a ninth folder
appears, which is exactly the owner's "choose which ones to include once the
subfolders do appear" case.

---

## 2. Arrival rule

Required because the producer writes non-atomically (§0.2). Two rules, cheapest
first:

1. **Stability:** a frame is complete once its size and mtime are unchanged across
   two consecutive polls.
2. **Neighbour evidence:** a frame is complete once a *later* index exists in the
   same camera folder (with rule 1 applied to that later file).

The last frame of an acquisition has no successor, so it is admitted by rule 1 plus
an end-of-acquisition sweep. That sweep has a natural trigger: the acquisition is
bounded and `metadata.json` appears at the end. **Two refinements the measured
acquisition forces:** `metadata.json` landed **6.0 s after the last frame**, so the
sweep cannot be the only admission path for the final frames — it must also fire when
every camera's count has been stable across N polls, or a session ends with its last
6 s of frames unadmitted. And a **short camera never gives its final frame a
successor at all** (§0.3.2): that frame is admitted by rule 1 alone, and the camera is
then marked excluded from interleaving rather than left dangling in the watermark. A
frame that never stabilises is quarantined and counted, never retried forever.

---

## 3. Incremental detection store

Purpose: make detection append-only, so a frame is detected at most once for the
whole session, and nothing needs re-deriving to answer "what do we have".

Constraints: identity is `(camera, acquisition, frame_index)` (§0.3.1);
`TargetDetection` is columnar `cam | global_im_num | key | x | y`
(`calibration_targets/core/target_detections.py:56`) and `__add__` concatenates
(`:251`) but requires equal `cam_names` and is not idempotent.

Suggested shape:

- one SQLite file in the stream working folder, one row per
  `(camera, acquisition, frame_index, key)` with x, y and a policy stamp;
- an append-only in-memory `TargetDetection` for the current rolling window, kept in
  `global_im_num` order for the solver;
- an in-memory index `{(camera, acquisition): sorted frame_index}` for the watermark
  and "what is missing" queries.

Why SQLite rather than extending the pickle idiom: that idiom's integrity contract
hashes the **whole** file on every validation (SHA-256 sidecar, `cache_matches`
:558), so extending it per frame costs O(total detections) per frame. A row store
makes the per-frame cost O(new observations). This is a performance requirement, not
a preference, and it is the only place this plan deviates from an existing repo
idiom.

Idempotence: a frame's observations are written once under its key and never
rewritten. If the policy stamp changes (detector backend, upscale factor, target spec
— the same things `detection_cache_name` encodes, `workflow/detections.py:132`),
affected frames are **re-detected and superseded**, not merged, so nothing is lost
and two detectors' output is never mixed.

---

## 4. Cadence

| Tier | Trigger | Work | Budget |
|---|---|---|---|
| 1 | every accepted frame | detect, append, cheap per-camera metrics | per-frame |
| 2 | every N frames or on a timer | bundle adjustment over a rolling window, warm-started | 1-5 s |
| 3 | minutes, or on demand | intrinsics re-selection, self-calibration, checkpoint | seconds-minutes |

- **Tier 1.** Sized at **one worker per camera (8)**, not at a thread count derived
  from a rate — see §0.1. The tick is a barrier, so the budget is the frame period
  (~200 ms at the rig's measured 5 fps) and `ceil(8/8) = 1` detection wave per tick,
  costing the slowest camera: 45-67 ms median, 67-99 ms p95 per camera, **134.25 ms
  worst measured single frame**. No subsampling policy is needed, but keep the
  configurable hook — a slower machine still wants it. Read one image, detect,
  append, release.
- **Tier 2.** Rolling window of the last W accepted frames (W ~ 20-60; at the rig's
  measured 5 fps a 30 s window is 150 frames, so W should be chosen from the *pose
  diversity* target and not from a wall-clock span) plus a small fixed set of
  **anchors** retained for the whole session, so the solve does not drift as the
  window slides. Initialize from the previous Tier 2 solution.
- **Tier 3, intrinsics.** `run_initial_calibration` is per-camera and independent
  (`camera_calibrator.py:209`); a camera needs *spread*, not volume. Keep a bounded
  per-camera selection (30-60 views chosen for hull area and pose diversity),
  recalibrate only cameras whose selection changed materially, and use the selection
  itself as the "covered yet?" signal.
- **Tier 3, self-calibration.** Explicit action only. The gauge subset is chosen per
  solve (`fixed_inds`, `visible_feature_mask`, `workflow/phase4.py:195-197`), so a
  growing observation set changes the gauge and the target shape would jitter while
  the operator watches. Gate it on `D4.7` mean target displacement (:215).

---

## 5. Metrics, visualisations, stop verdict

| Requirement | Source |
|---|---|
| Per-camera detection heatmap (sensor space) | `D1.6_spatial_coverage` + `D1.4_features_matrix` (`workflow/phase1.py:408-418`; `_spatial_coverage` :430) |
| Per-camera RPE | `per_camera_mean_reprojection` (`workflow/diagnostics.py:43`) |
| System mean Euclidean RPE | norm over `observation_residual_xy` (`diagnostics.py:20`) — **must use the reshape guard at `diagnostics.py:34`**: lockbox prior residuals are appended after the reprojection residuals and contaminate a naive norm (crashes on odd camera counts) |
| Target reconstruction | `optimisation/find_target.py`; `CameraSet.get_mesh()`/`get_viewcone()`; Open3D via the out-of-process viewer path (`gui/viewer_process.py`) |

1. **"Heatmap" is ambiguous.** Sensor-space coverage (2D histogram) is cheap and
   live; volume-space coverage needs solved poses (Tier 2). Recommend both:
   sensor-space heatmap at Tier 1, target-position cloud at Tier 2.
2. **RPE must not be the primary "good enough" signal.** It can look excellent while
   the rig geometry is wrong — every camera consistently wrong agrees with itself. The
   primary signal is joint-observation span and target pose diversity. Report RPE per
   camera with its observation count, and read a camera below a view threshold as
   "insufficient evidence" rather than as a number.
3. **Deliver a stopping verdict, not a dashboard:** per-camera coverage target plus an
   RPE floor, combined into a "you can stop now" flag, so the operator is not left
   reading numbers while holding the board.

---

## 6. Checkpoint, ownership, failure isolation

**Checkpoint.** A run has a start, an end, an id and artifacts
(`workspace.save_run`, `workflow/workspace.py:222`); a session does not.
`StreamSession.checkpoint()` materialises the accumulated detections as a run-local
`detected_datapoints.pickle` (the shape `save_detections` already writes), the current
camset as a `.camset`, and normal `phase1`/`phase2`/`phase3` run records through
`WorkspaceManager` — so Phase 2/3/4 tabs, Assess Calibration and the export path
consume it with no knowledge that it came from a stream. Without this the feature is a
parallel universe; with it, stopping the session immediately yields an ordinary
calibration the operator can refine or export.

**Ownership.** The stream writes its own subdirectory and never shares a run slot or
the image-folder cache name with a conventional Phase 1 run. Note that the images and
`metadata.json` live in one experiment folder that is still being written to by the
acquisition while the stream reads it.

**Failure isolation.** The existing runners catch errors only at top level
(`phase1.run`'s `_detect` wrapper). A session needs finer containment: skip and count
a corrupt or half-written frame; mark a stalled camera degraded and continue with the
rest; on a Tier 2/3 solve failure keep the last good solution and retry at the next
boundary; re-detect in the background after a policy change while Tier 1 keeps
appending. All of it must surface as visible state, not terminal output — the owner's
stated pain is having to watch logs.

---

## 7. Staged implementation

Each stage is independently verifiable and leaves the repo working.

- **S0 — measure on real data.** One real calibration acquisition at 720x540 Mono16,
  20 ms. Confirm the on-disk layout and the arrival behaviour; time detection per
  frame; confirm `metadata.json` appears only at the end. Deliverable: a number and a
  folder listing, not code. **DONE** — `pyCamSet_paper/ccube2/5_corners/experiment_001`,
  797 frames, 8 cameras, measured 4.95 fps (not 2), target Ccube2 6 pts / 10 mm;
  report at `D:\Work\coding\reconstruction\_s0_measurement_ccube2\`.
  `[VERIFIED-BY-EXECUTION — 1336 passed, 1 skipped on the control-software branch]`
- **S1 — store + watcher, backend only.** `stream_store.py` + `stream.py`, no GUI:
  feed it a folder as frames appear; prove append-only behaviour and idempotence with
  a headless test. **Do not build that test on the ChArUco fixtures** — the rig flies
  **Ccube2 6 pts / 10 mm** (§0.2), so the fixtures prove a path the rig does not run.
  Render the Ccube2 board through pyCamSet's own target class, as the S0 tool's tests
  already do (`tests/test_detection_cost_measure.py:47` renders through
  `detection_worker._build_charuco2_target()`), and keep the ArUco 1 ChArUco fixtures
  only as a fast synthetic shape check.
- **S2 — Tier 1 metrics + stop verdict.** Coverage, detection rate, completeness,
  per-camera status. Headless-verifiable.
- **S3 — GUI tab (subscriber only).** Folder picker, four non-latching policy groups,
  live Tier 1 panels. Headless-testable per `test_gui_phase_contracts` conventions.
- **S4 — Tier 2.** Rolling-window BA with warm start and anchors; per-camera and
  system RPE; target reconstruction.
- **S5 — Tier 3 + checkpoint.** Bounded intrinsics selection, explicit
  self-calibration, run-emitting checkpoint.

Each stage must add a headless test that fails if GUI-only coupling appears.

---

## 8. Owner decisions and remaining questions

Answered 2026-09-19, recorded here so they are not re-litigated:

1. **Capture geometry:** 720x540 Mono16 during calibration (full frame, not the
   540x440 experiment ROI). **Exposure:** 20 ms. **Rate:** the owner's ~2 fps is
   **superseded by measurement** — the only calibration acquisition on the rig ran at
   **4.95 fps** (metadata `Frequency: 5`, sequence `target.frequency: 5`, 20.0 s over
   99 intervals), and no 2 fps acquisition exists on disk to support the figure. The
   plan is now rate-independent by construction (§0.1), so this is recorded rather
   than re-decided: the pool is sized at the camera count, and the tick budget is
   whatever period the operator sets.
2. **Index correspondence:** **no longer an owner assumption.** Measured: the cameras
   start at index 0 and run contiguously; a short camera is tail-truncated, so its
   written frames keep their true indices. Tier 1 guard = identical index start +
   contiguity, short camera excluded from interleaving, never offset-corrected
   (§0.3.2). Per-frame *simultaneity* is not provable from a finished folder at all —
   that needs `(block_id, timestamp)` persisted by the producer, which is now the
   recorded durable fix rather than an assumption to carry.
3. **Rig static:** yes — Tier 2 is a refinement, and extrinsics are not expected to
   solve from nothing.
4. **Warm-start baseline:** deferred. Consequence: the session must not *require* a
   prior calibration to function. Tier 2 starts from the Tier 3 intrinsics solution
   plus the first window's BA; a supplied camset becomes an optional accelerator added
   later (tracked on the follow-up card, §9).
5. **Session unit: case A first.** One session = one bounded acquisition. Multi
   acquisition concatenation (case B) is **not** in the first implementation pass and
   is tracked on the follow-up card (§9).
6. **High-distortion redetect path:** explicitly out of scope for the initial
   real-time functionality (tracked on the follow-up card, §9).

Still open:

1. **S0 measurement — CLOSED** (§7). Measured on the rig's own acquisition; the
   numbers, the four defects it exposed, and the Ccube2/rate/index findings are in the
   evidence journal (`m_nebula_microcontroller/AGENTS/references/evidence-journal.md`,
   2026-09-19 entries) and the report at `_s0_measurement_ccube2/`.
2. **Target-parameter source.** With `metadata.json` written only at the end, §0.2
   assumes you select the target in the pyCamSet tab and let it cross-check against
   `metadata.json` when that appears. Confirm, or say what you would rather do.
   **New input, measured:** that cross-check has *never* run on this rig and cannot
   yet — `Calibration_Targets.Targets` is empty in **every** `metadata.json` on disk
   (only two files carry the key at all, both `[]`). It is fed by
   `ExperimentService.set_calibration_targets` (`experiment_service.py:273`,
   `:1522`), called only from `ExperimentPanel._apply_sequence`
   (`ui/experiment_panel.py:1459`) for rows the operator explicitly added as
   **Calibration** rows — which are host-only and deliberately kept off the P2 wire.
   So enabling it is an operator step (add a Calibration row to the sequence), not a
   code change; until an acquisition carries one, the cross-check stays unexercised
   and the tool's "no Calibration_Targets entries" warning is correct behaviour.
   `[VERIFIED-BY-READING]`

---

## 9. Deferred follow-up work

Tracked on Kanban card **`t_9b98e9a9`** (board `work`), so none of it is forgotten:

1. **Streaming across multiple acquisitions (case B).** Concatenate consecutive
   acquisitions in one folder; coverage and the stop verdict become per-session.
2. **Warm start from an existing calibration** — optional accelerator, never required.
3. **pyCamSet reads the calibration target from the images' metadata**, so the user
   does not define the target in two places. Note `metadata.json` is written only at
   acquisition end (`experiment_service.py:633`), so it cannot drive live detection as
   things stand.
4. **High-distortion redetect path** (`detect_datapoints_in_imfile(..., camset=...)`).
5. **metadata/PyCamSet target cross-check** (owner, 2026-09-19): pyCamSet compares its
   configured target against `metadata.json` and **reports and refuses to proceed** on
   a mismatch, rather than calibrating against the wrong geometry. For the first pass
   the operator defines the target in pyCamSet and nothing reads the metadata; the
   control software's `metadata.json` write timing is the constraint (item 3).
6. **Persist per-frame `(block_id, timestamp)` in the producer.** Prompted by the S0
   run (§0.3.2): no per-frame time reaches disk today, so a finished folder cannot
   prove that index *i* is the same instant in two cameras, and the *only* artifact
   that can prove it — the camera's own `GetBlockID()` — is already being read into a
   log line and thrown away. `services/simultaneity_analysis.py` already knows how to
   pair on it (54 ns residual on real hardware versus a 20.5 ms apparent skew from
   index alignment). Writing it next to each frame, or into a per-acquisition sidecar,
   converts the stream's index assumption into a check. This belongs to the control
   software, not pyCamSet — @rebels and I split it that way: the design doc is mine,
   the producer change is the rig side.
7. **The S0 tool's `expected_for_comparability` constant — SUPERSEDED by §10.1.**
   It named `d6e2c6da`, so every report warned against its own checkout. The
   constant was then repointed at a commit pair, which failed for the same reason
   (see §10.1), and the whole commit-based gate has since been **removed**: the
   commit pair is now provenance only and deliberately produces no warning.
   Comparability is the measurement module's own hash. Recorded here so the
   history of that decision is not lost, not as an open item.
8. **`environment.yaml`'s trailing `prefix:` line.** The file ends with a tracked
   `prefix: C:\Users\MSOT\.conda\envs\panoramic-control-refactored` — an absolute
   machine path one line below the comment explaining why absolute machine paths do
   not belong in that file. It is a conda `env export` artefact and harmless to
   `conda env create -f`, but it is the same class of machine-specific coupling the
   S0 plumbing fix just removed, so it should be dropped or commented out.

Case B and the redetect path were previously listed as open questions; they are now
decided (deferred) rather than open, and the answers are recorded in §8.

---

## 10. Addendum — S0 re-run after the target-layer work (REBELS, 2026-09-20)

Item 7 above is done, and the owner settled two of the remaining questions. Both
changed a constant, and one of them removes the basis the plan's thread verdict
was computed on, so it is recorded here rather than edited into §0.1.

### 10.1 Comparability is decided by the module hash, not a commit pair

The S0 report pinned its expected commits, first at `d6e2c6da` and then at the
commit that made the change. Both were wrong for the same reason, and the second
made it obvious: **a constant that names the commit it lives in cannot exist
before that commit is made**, so any commit-based gate is either stale on the
checkout it was written for or needs updating after every commit -- including
commits that change nothing the measurement does.

So comparability is now the measurement module's own SHA-256, read from the file
at run time: two machines measured the same code iff it matches. The commit pair
is still reported as a provenance record, and the mismatch **warning is gone** --
it fired on every correct run on this branch, and a warning that fires when
nothing is wrong is a warning nobody reads.

The work is on **`main`** at `4e602f3d` (both remotes; MSOT updates it via
PyCharm). Comparability is the **module hash**, and it is normalised for line
endings — see §10.6. Do not run `d6e2c6da`, which contains the four defects
§0.1's numbers were computed through. `[VERIFIED-BY-EXECUTION — 1364 passed,
1 skipped]`

### 10.2 The rate became a setting, which supersedes §0.1's thread verdict

§0.1 concludes "the rate is not ours to choose, so the pool must not depend on
it" and sizes the Tier 1 pool at one worker per camera. The owner's position is
the opposite in kind, not in effect: **2 fps is the intended general rate**
("I might acquire images at different framerates but I thought ~2 fps was a good
rate to use generally"). So the rate is a *chosen* setting, not a discovered
constant, and the S0 tool now takes it as an input (`Acquisition rate
(fps/camera)` and `Cameras` in the dialog) instead of baking in 16 frames/s.

Re-measured on the same folder (320 frames, 8 cameras, Ccube2 6 pts / 10 mm,
40 frames per camera):

| rate | `worker_threads_if_serial_median` | `..._p95` |
|---|---|---|
| 2 fps/camera (16 frames/s) | 1 | 2 |
| 10 fps/camera (80 frames/s) | 4 | 6 |

Cost was unchanged between the two runs (end-to-end median 50.11 vs 48.44 ms),
as it must be — only the rate the answer is computed for differed. The §0.1
verdict (one worker per camera, tick as a barrier) is unaffected: it was derived
from the *cost*, not from the rate, and the cost reproduces.
`[VERIFIED-BY-EXECUTION — `_s0_real_2fps` / `_s0_real_10fps` report folders]`

### 10.3 The Ccube's `length` is invisible to detection — a real gap in §0.2's check

§0.2 expects a configured-vs-physical mismatch to be caught. **It cannot be
caught through detection for this target.** Measured: 32 real frames across 4
cameras, Ccube2 at `length=10.0` and at `length=30.0`, give **byte-identical
detected corner coordinates on all 32** (and identical counts on a 12-frame
earlier probe). The face's marker layout is a function of `n_points` and the
dictionary, not of the printed edge, so a 3x wrong cube size detects exactly as
well as the right one.

Consequences, and they are the actionable part:

1. The S0 tool's target cross-check is only as strong as the *declared* spec.
   On `experiment_001`, `metadata.json` carries `Calibration_Targets: []`
   (empty), so the cross-check could not run and the report says so instead of
   implying a pass. Item 6's producer-side work should therefore include
   **writing the target spec into `metadata.json`'s `Calibration_Targets`**, not
   only BlockID: without it the geometry check has nothing to check against.
2. `n_points` is the field that *is* visible to detection — a wrong square count
   changes the marker layout and the corner ids. A physical check of the cube's
   size has to come from a measurement (calipers, or a known-distance
   reconstruction), not from detection.

### 10.4 Detections on the rig's real frames are not clean

Measured, and not diagnosed here: **1-4 of every 40 frames per camera find no
corners at all** (cam2/cam6/cam8 worst at 4/40), and aruco2 emits repeated
`Marker N corner M connected to marker P corner Q but global corner ids differ`
warnings on these frames. Detection cost tracks the corner count almost
linearly — cam5/cam6/cam8 are cheapest (~38-41 ms median) and carry the most
no-detection frames, cam1/cam4 dearest (~55 ms) and the fewest. §0.1's numbers
were computed over exactly these frames, so a p95 of 72-84 ms is a p95 *of*
a population that includes frames where nothing was found. Whether that is a
board too small in frame, motion, or the corner-id disagreement aruco2 is
warning about is open, and it belongs to the rig side.
`[VERIFIED-BY-EXECUTION — per-camera no-detect counts in
`_s0_real_2fps/detection_cost_report.json`]`

### 10.5 Target definitions are now read from pyCamSet

Requested by the owner, and the check the request was based on came out
negative: pyCamSet's restructure did **not** break this repo's imports. All
seven pyCamSet import paths the control software uses still resolve at
`upstream/development` (`9b44b06`, 54 commits past `b79bb76`), and at
`upstream/master`; the files moved *within* the package but kept their dotted
paths. `environment.yaml`'s pyCamSet line was already corrected in the earlier
S0 commit.

What was added instead is the guard: `services/pycamset_targets.py` reads
pyCamSet's own registry for the target list, each target's arguments and
whether pyCamSet will build a given set of values; `probe_import_paths()` checks
all seven paths and, where a symbol has moved, searches pyCamSet for its new
location and names it; and the Define Target dialog states which pyCamSet it is
validating against. The rig's own values stay local — a library default is not a
rig measurement (pyCamSet defaults a Ccube to 20 mm; the rig's cube is 10 mm, per
§0.2 and the owner). `[VERIFIED-BY-EXECUTION — 25 tests, plus the live probe]`

Item 8 above (the `prefix:` line) is untouched: it is a cosmetic export artefact
in a file the branch already edits, and it is M_NEBULA's file to change.

### 10.6 The comparability hash was platform-dependent — corrected

§10.1 replaced the commit pair with the measurement module's SHA-256. That hash
was computed over the file's **raw bytes**, which makes it platform-dependent:
git's `core.autocrlf=true` (the Windows default) writes LF as CRLF on checkout,
so one commit has different bytes on disk on Windows and on Linux. Found by
M_NEBULA and reproduced here before changing anything, on the same commit:

| tree | sha256 | CRLF pairs |
|---|---|---|
| Windows worktree, as the function read it | `e1a7ad61…9855cdc8` | 1345 |
| LF-normalised (== the committed git blob) | `3cf2719b…b770e958` | 0 |

So a fully correct pair of runs would have read as a mismatch — the same
false-alarm failure the commit-pair check had, moved into the field that is now
the only arbiter. The hash reduces CRLF to LF before hashing, which makes it
invariant across Windows/Linux/macOS and equal to the committed blob. Regression
tests assert both trees yield one value, that the value equals the git blob, and
— as a negative control — that the previous raw-bytes logic is **not** invariant
(LF `5cbef7d7…` vs CRLF `1ed3e8a4…`), so the test can actually fail.

The repository also gained a `.gitattributes` (`* text=auto eol=lf`, binaries
marked `binary`), because the underlying cause is that the repo had none: the
index was already LF but every Windows worktree was CRLF, so any future
byte-level comparison would have hit the same trap. No mass renormalisation:
`git ls-files --eol` shows 115 files CRLF in the worktree and LF in the index, and
adding the file leaves the tree clean apart from the edited files.
`[VERIFIED-BY-EXECUTION — reproduced both hashes, negative control run, 1364
passed, 1 skipped]`

