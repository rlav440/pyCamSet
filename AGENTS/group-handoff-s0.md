# Group handoff — pyCamSet real-time calibration, first task (S0 measurement)

Goal: we are adding **real-time (streaming) calibration** to pyCamSet, developed
against the M_NEBULA rig. Full design is already written:
`D:\Work\coding\reconstruction\pyCamSet\AGENTS\realtime-calibration-plan.md`
(read it — plan v0.2). **Do not redesign it.** The first task is only S0: a
measurement, because the plan's cadence constants depend on a number nobody has.

REBELS owns pyCamSet; M_NEBULA owns the rig, the P2 firmware and the control
software. Work the task together, and say plainly which of you verified what.

---

## State as of this handoff (both repos are pushed and clean)

| Repo | Branch | Commit | Remote |
|---|---|---|---|
| pyCamSet fork | `development` | `b79bb76` | pushed to `origin` = `github.com/ColDSnit/pyCamSet.git` |
| Control software | `main` | `d6e2c6da` | pushed to `github` = `github.com/ColDSnit/software-panoramic-refactored.git` |

`b79bb76` = "Say what the never-seen points actually are" (76 commits past the
previous remote tip). The control software's two new commits are:

- `8631bb09` `fix(detection): follow pyCamSet's calibration_targets module move` —
  the worker imported `calibration_targets.target_charuco`, which no longer
  exists; it is now `calibration_targets.charuco.target`. Without this the worker
  cannot be imported against the current pyCamSet at all.
- `d6e2c6da` `feat(targets): offer pyCamSet's ArUco 2 boards, ChArUco2 and Ccube2`
  — the new targets, selectable in the Define Target dialog.

**The control-repo tree is clean**; the previously uncommitted import migration is
in `8631bb09`. Nothing is left dangling.

---

## Absolute paths (this PC)

| Thing | Path |
|---|---|
| pyCamSet fork (branch `development`) | `D:\Work\coding\reconstruction\pyCamSet` |
| The plan (read first) | `D:\Work\coding\reconstruction\pyCamSet\AGENTS\realtime-calibration-plan.md` |
| Control software (branch `main`, PyQt6) | `D:\Work\coding\reconstruction\coldsnit\control_software\software-panoramic-refactored` |
| Env for the **control software** (pyCamSet + PyQt6 6.11.0 + cv2) | `D:\ProgramData\anaconda3\envs\panoramic-control-refactored\python.exe` (Python 3.12.13) |
| Env for **standalone pyCamSet** work (a real, working env) | `D:\ProgramData\anaconda3\envs\calibration_15092026\python.exe` |
| ArUco 2 board parameters, both sides | control software `models/calibration_target.py`; pyCamSet `calibration_targets/charuco2/target.py` |
| Follow-up Kanban card (board `work`) | `t_9b98e9a9` |

**Note on the env you may have been told to use:** `calibration_07032026` is
**broken** — a Python 3.13 interpreter with cp311 numpy binaries, so `import numpy`
fails with `No module named 'numpy._core._multiarray_umath'`. Do not run pyCamSet
tests there. `calibration_15092026` imports numpy and cv2 cleanly.

The control repo's `environment.yaml:39` installs pyCamSet editable from the
pyCamSet path above, which is why the two live side by side here.

Two envs, deliberately: the control repo's own env is the one that must run its
tests (`QT_QPA_PLATFORM=offscreen … -m pytest tests/ -q`, currently **1286 passed,
1 skipped**). Do not measure detection cost in an env the GUI does not use.

**aruco2 is installed in `panoramic-control-refactored`** (built from the submodule,
same route as MSOT). It is *not* in `environment.yaml` yet — that is the one open
plumbing item, and it is now **required rather than optional**, since the rig uses
an ArUco 2 board.

---

## The task: build a detection-cost measurement tool, in the GUI, and push it

Make it part of the **software-panoramic-refactored** GUI. I will pull on MSOT (the
machine wired to the rig) and run it there on real data.

**What it must measure**, on real calibration-acquisition frames (720x540 Mono16
`.tiff`), pointed at a folder of images — it must not need live cameras to run:

1. End-to-end detection time per frame: median and p95, per camera, for the rig's
   ChArUco board. This is the number the plan is waiting for.
2. The two input paths, separately: (a) raw Mono16 as read from disk, (b) the
   contrast-stretched uint8 path the GUI already uses. I need to know whether the
   stretch costs anything meaningful.
3. TIFF decode time alone, so I can separate read cost from detection cost.
4. Threads used per detection call, and therefore how many worker threads 16
   frames/s (8 cameras x 2 fps) actually needs.

**Reuse, do not reinvent.** `services/detection_worker.py` already imports the real
pyCamSet targets and calls `find_in_image()`:

- `_build_charuco_target`, `_build_ccube_target` — the rig's ArUco 1 targets
- `_convert_uint16_to_uint8` — the existing Mono16 → uint8 contrast stretch
- `DetectionWorker`, `_detect_charuco`

Line numbers drift as the file changes — grep for the names rather than trusting
offsets. Measure the path the GUI actually uses; a faster measurement of a path we
do not use is worthless. Keep the existing tests passing.

**What it must write out:** a JSON (and a short human-readable summary) with the
numbers above, plus the folder listing it ran on, the filenames it saw, the target
spec it detected with, and the host/CPU it ran on. Save it somewhere findable and
tell me the path.

---

## The S0 questions it must also answer (cheap to record while it runs)

1. Confirm the on-disk reality: camera subfolder names, the
   `{camera}_{YYYYmmdd_HHMMSS}_{frame_index:06d}.tiff` pattern, and whether
   `metadata.json` is absent during the acquisition and present after it ends.
2. Confirm the rig's actual ChArUco parameters used at acquisition, and that they
   match what pyCamSet is configured with. A configured-vs-physical mismatch has
   already cost ~7 wasted investigation rounds on this project — **report a
   mismatch, never proceed through one.**
3. Record how long after acquisition start the first frame lands, and whether the
   file count grows smoothly.

---

## Decisions already made — do not re-litigate these

Carried from the plan and the owner; they are settled.

- **Capture**: 720x540 Mono16 at ~2 fps, 20 ms exposure. 8 cameras. The 540x440
  figure is the mouse-heart *experiment* ROI and is **not** the calibration ROI.
- **Detection load is small**: 16 frames/s total. The earlier "~31 cores" figure
  came from the 387 fps experiment rate and does not apply. Do not design a
  subsampling scheme for this.
- **Only `metadata.json` at acquisition end**: `ExperimentService._save_metadata`
  runs on the success path after the firmware tail wait, so metadata cannot drive
  live detection. The operator defines the target in pyCamSet for now; reading the
  target from metadata is deferred to card `t_9b98e9a9`.
- **No calibration mode exists.** `set_calibration_mode` / `configure_calibration`
  remain reachable programmatically but have no UI caller; frames land in the
  normal auto-generated experiment save path. Do not build against the calibration
  save path.
- **Session = one acquisition, case A.** Multi-acquisition streaming (case B) is
  deferred to the card.
- **High-distortion redetect path** (`detect_datapoints_in_imfile(camset=...)`) is
  out of scope, deferred to the card.
- **The rig uses an ArUco 2 board** (owner, 2026-09-19). So the S0 measurement must
  detect with the ArUco 2 path — `_build_charuco2_target` / `_detect_charuco2`
  (ChArUco2), not the ArUco 1 `_detect_charuco`. Measuring the ArUco 1 path would
  produce a number for a detector the rig does not run. Confirm on the bench which
  ArUco 2 target it is (ChArUco2 board vs Ccube2 cube) and measure that one; if
  both are in use, measure both.

---

## Binding constraints (from this repo's own briefs)

- No PyQt6-free reimplementation: the measurement is GUI-integrated, as asked.
- Do not touch Spin2 firmware or anything electrically consequential.
- Do not add a dependency without checking `environment.yaml` first.
- Cross-platform by default: `pathlib`, UTF-8, no hardcoded `D:\` paths in
  application code, no OS-specific shell commands. MSOT is a different machine.
- Preserve existing tests and headless behaviour; `mock` mode must keep working.
- Keep GUI wiring and backend logic separable, as the rest of this repo does.
- Conventional commits (`feat(scope): …` / `fix(scope): …`), as the history uses.

---

## Git and MSOT (read before pushing)

- Remotes on the control repo: `github` =
  `github.com/ColDSnit/software-panoramic-refactored.git`; `origin` =
  `https://www.iekm.uniklinik-freiburg.de/gitlab/kok/software-panoramic-refactored`.
  **Push to `github`**, tell me the branch name and commit SHA, and read the pushed
  ref back with `git ls-remote` to confirm it landed. Work on a branch, not `main`,
  unless you have a reason to say otherwise.
- **MSOT must run the same pyCamSet and control-software commits as this PC, or the
  measurement is not comparable.** Record both SHAs (`b79bb76`, `d6e2c6da`) in the
  report and in your message to me.
- **Open item — the editable pyCamSet install path.** `environment.yaml:39` pins
  `pip install -e D:/Work/coding/reconstruction/pyCamSet`, and **that path will not
  exist on MSOT.** Either MSOT mirrors that exact path, or the install step becomes
  configurable. Pick one, write down which, and tell me — do not let me discover
  this at the rig. This is the one piece of plumbing I want settled before the run.
- **`aruco2` is installed on MSOT** (`pycamset` env), built from the
  `third_party/aruco2` submodule — it is not on PyPI and has no published wheel.
  The build needs all three of: MSVC Build Tools, a native OpenCV dev package, and
  `OpenCV_DIR` pointed at a directory that exists. On MSOT the working combination
  was conda-forge `opencv=4.9.0` (whose CMake files live in
  `$CONDA_PREFIX/Library/cmake/x64/vc16/lib`):

  ```bash
  pip install -v . --config-settings=cmake.define.OpenCV_DIR="$CONDA_PREFIX/Library/cmake/x64/vc16/lib"
  ```

  Pointing `OpenCV_DIR` at the parent `Library/cmake/` **fails**: that config derives
  the runtime from the CMake *generator*, and with an NMake/JOM generator every
  fallback rung fails, ending in `OpenCV_FOUND=FALSE`. Do not re-derive this.

---

## Watch out for

- **pyCamSet's codegen cache.** `pyCamSet/optimisation/template_functions/*.py` are
  *generated* (`abstract_function_blocks.py`, `matmul_map.py` write them), tracked
  in git, and they regenerate on first solve. Three untracked telecentric ones exist
  in the local checkout and were deliberately **not** pushed. Never hand-edit them.
- **Frames are rotated 90° clockwise before saving** (`_rotate_frame`,
  `camera_service.py`), so detection and any sensor-space heatmap are in rotated
  coordinates. Do not report a coverage/heatmap result without saying which frame
  it is in.
- **The producer writes frames non-atomically** (`cv2.imwrite` straight to the final
  filename). If the tool watches a live acquisition rather than a finished folder,
  a half-written frame is possible. S0 reads a finished folder, so this should not
  bite — but do not "improve" the tool into a watcher without handling it.

---

## What good looks like for this task

A user-visible way to point the tool at a folder of real frames and run it, a JSON
report on disk, the existing test suite still green, a pushed branch with a SHA, and
a short message back to me with the actual numbers. The numbers are the deliverable —
the plan cannot set its window sizes and thread counts without them.
