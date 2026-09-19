# MSOT runbook — real-time calibration S0 measurement

You said you would update both repos on MSOT and then test. This is the exact
sequence, in order, with what to check at each step. Nothing here needs a rig
connected: the measurement reads a folder of already-acquired frames.

---

## What you are starting from

| Thing | Where |
|---|---|
| Control software branch | **`main`** (the owner's channel; updated via PyCharm, not `git pull`) |
| Control software commit | **`4e602f3d`** (or later on `main`) |
| pyCamSet branch/commit | `development` @ **`b79bb76`** (unchanged) |
| Reference measurement folder | `E:\M_Nebula\1 Data\1 Data\pyCamSet_paper\ccube2\5_corners\experiment_001` |

**`main`, not the feature branch.** The work is on `main` on both remotes, so
that is what MSOT should be on: `git ls-remote origin refs/heads/main` should
read `4e602f3d`. The owner updates the checkout through PyCharm to keep local
machine-specific changes (mock mode), so `main` is the channel that works for
them — do not send them to a branch or a Bash `git pull`.

**Do not run `d6e2c6da`.** That commit contains four defects the measurement has
since fixed — most importantly, it silently reports no thread answer on any real
folder.

**Comparability is checked by hash, not by commit.** The measurement module
hashes *its own source*, line endings normalised, and the summary prints it:

```
module sha256   : 5cbef7d76e77f7bc9e900eaf18c05b76b83a6901c8777c74a15a8e1e0aa7bb42
```

That is the value this revision prints. Compare it against MSOT's summary; it
should read the same there.

The two machines measured the same code iff these match. The commit pair appears
in the summary too, as a record of provenance, and deliberately produces **no**
warning when it differs — a commit SHA cannot name the commit it lives in, so it
cannot be the check.

**Line endings are normalised before hashing, and that is load-bearing.** Git's
`core.autocrlf=true` (the Windows default) writes CRLF on checkout, so raw bytes
differ between Windows and Linux for one commit. The hash reduces CRLF to LF
first, so it is platform-invariant, and the repository now carries a
`.gitattributes` (`* text=auto eol=lf`) so the working tree is LF everywhere too.
If you ever see a hash mismatch between two machines on the same commit, check
this first.

---

## Step 1 — get both repos on MSOT

Control software: **on `main`.** The owner's channel is PyCharm, not a Bash
`git pull`, so pull/update `main` there rather than checking out a branch.

```bash
cd <your checkout of software-panoramic-refactored>
git ls-remote origin refs/heads/main     # expect 4e602f3d
git rev-parse HEAD                       # expect 4e602f3d (or later on main)
```

pyCamSet:

```bash
cd <your checkout of pyCamSet>
git checkout development
git pull                     # expect b79bb76 at tip, or note what you got
git rev-parse HEAD
```

**The commit SHAs are a record of provenance, not the comparability check.** The
report records both and deliberately does **not** warn when they differ. What
decides comparability is `measurement_module_sha256`, printed in the summary:
two machines measured the same code iff those match. See the header above for
why line endings matter to that value.

## Step 2 — confirm the environment can see pyCamSet

```bash
conda activate panoramic-control-refactored
python -c "import pyCamSet, cv2, numpy, tifffile, aruco2; print('pyCamSet', pyCamSet.__file__); print('cv2', cv2.__version__); print('aruco2 OK')"
```

If `aruco2` is missing, neither ArUco 2 target can be built and the measurement
cannot run — that is the one hard prerequisite. `docs/ENVIRONMENT_SETUP.md` in
the control repo has the build (conda-forge `opencv=4.9.0` + `OpenCV_DIR`
pointed at `$CONDA_PREFIX/Library/cmake/x64/vc16/lib`; pointing it at the parent
`Library/cmake/` fails).

Check that pyCamSet's target API is where this software expects it:

```bash
python -c "import sys; sys.path.insert(0,'.'); import services.pycamset_targets as t; h=t.import_health(); print('ok:', h['ok'], '| broken:', [b['label'] for b in h['broken']]); print(h['python_package'])"
```

Run from the control-software checkout. `ok: True` means all seven pyCamSet
import paths this software uses resolve. If any are broken, the output names
where the symbol moved to, and that is the line to send back.

## Step 3 — run the test suite (the environment sanity gate)

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q
```

Expect **1362 passed, 1 skipped**. A different count means the environment
differs from the one this was developed and measured in, and the measurement
numbers would not be comparable either.

## Step 4 — run the measurement

### Through the GUI

1. `python main.py` (mock mode is fine — cameras are not used).
2. **Camera Control → Measure Detection Cost**.
3. Frames folder → the reference folder above (or a fresh acquisition).
4. Target type **Ccube2**, squares/face **6**, edge **10.0 mm**. Choosing the
   folder pre-fills these from the folder's own `metadata.json` where it carries
   them; on `experiment_001` it carries `Calibration_Targets: []`, so fill them
   in by hand.
5. Acquisition rate **2 fps/camera**, cameras **8** — these set what the
   worker-thread answer is computed for, not the measurement itself.
6. Frames per camera: `0` = every frame. For a first pass, 40 is enough to see
   the shape of it.
7. **Run Measurement**. The panel shows the numbers; the JSON and a summary are
   written to `<frames folder>/detection_cost_measurement/`.

### From a shell (same thing, no GUI)

```bash
python -c "
import pathlib, services.detection_cost_measure as m
opts = m.MeasurementOptions(
    folder=pathlib.Path(r'E:/M_Nebula/1 Data/1 Data/pyCamSet_paper/ccube2/5_corners/experiment_001'),
    out_dir=pathlib.Path(r'D:/Work/coding/reconstruction/_s0_measurement_ccube2'),
    target_type='Ccube2', ccube_n_points=6, ccube_length_mm=10.0,
    ccube_border_fraction=0.1,
    expected_fps=16.0,
)
rep = m.measure_folder(opts)
print(open(rep['written']['summary'], encoding='utf-8').read())
"
```

Run it from the control-software checkout so `services` resolves.

## Step 5 — what to check in the output

Send back the summary file. The lines that matter:

- **`end-to-end` median and p95** — the number the plan is waiting for.
- **`detection median` all-threads vs `one_opencv_thread`** and
  `internal_parallel_speedup` — whether one detection uses more than one core.
- **`worker_threads_if_serial_median` / `_p95`** — the thread answer for the
  rate you entered.
- **Warnings.** Every one is there for a reason. In particular:
  - `metadata.json carries no Calibration_Targets entries` means the
    configured-vs-physical check could not run — **the run did not verify the
    cube's geometry**, and on `experiment_001` that is expected;
  - a `.pycamset_workspace` note is normal and not a problem.

There is deliberately **no** commit-mismatch warning. Compare
`measurement_module_sha256` between the two machines' summaries instead.

---

## What this run does NOT do

Stated plainly so nothing is over-read:

1. **It does not detect stream calibration.** No streaming code exists yet — the
   plan is at S1 (store + watcher, backend only). This run measures detection
   cost on a finished folder, which is what sets the plan's constants.
2. **It does not verify the cube's physical size.** A Ccube's printed edge is
   invisible to detection: 10 mm and 30 mm give byte-identical detections on the
   rig's own frames (32/32 frames, 4 cameras). A wrong cube size detects exactly
   as well as the right one. Until `metadata.json` carries the target spec, that
   check cannot run at all.
3. **It does not check simultaneity.** The folder's own timestamps cannot show
   it — the volume is exFAT, whose mtime granularity is 2 s against a ~200 ms
   frame period, so the interframe gap reads 0 ms median / 2000 ms p95 and the
   tool withholds its verdict rather than calling that "smooth". Proving index
   *i* is the same instant in two cameras needs the camera's `BlockID`
   persisted next to the frames, which the producer does not do yet.

## The reference numbers, for comparison

From this PC, `experiment_001`, 320 frames (40 per camera), Ccube2 6 pts / 10 mm:

| | value |
|---|---|
| end-to-end | median 50.11 ms, p95 72.40 ms |
| TIFF decode | 2.61 ms |
| contrast stretch | 1.05 ms |
| detection, all threads | median 46.30 ms |
| detection, 1 thread | median 52.87 ms |
| internal speedup | 1.14x |
| worker threads @ 2 fps/camera (16 frames/s) | 1 median, 2 p95 |
| worker threads @ 10 fps/camera (80 frames/s) | 4 median, 6 p95 |

MSOT's numbers will differ — different CPU, possibly different OpenCV build.
That is expected and is exactly why the run exists on that machine: what has to
match is the **module sha256**, not the commits, so the two machines are
measuring the same code. A commit can differ (a later commit that changes
nothing the measurement does) while the measurement is still identical; the
hash is what decides it.
