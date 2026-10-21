# Contributing to pyCamSet

Thanks for taking the time. This page is the short version: what to install, what to
run, and what a pull request needs to clear. Everything here applies whether you are
contributing by hand or with help from a tool.

## Getting set up

```bash
git clone https://github.com/rlav440/pyCamSet.git
cd pyCamSet
pip install -e ".[dev]"
```

`[dev]` adds pytest and its plugins. Other extras: `optimisation` (optuna), `viz`
(open3d), `puzzle` (the PuzzleBoard targets), `docs` (mkdocs), and `all` for the lot.
PySide6 is a base dependency — the GUI is part of the default install.

The ChArUco and Ccube targets need a native Cairo library that pip cannot install.
On conda, `conda install -c conda-forge cairo`; otherwise see
[docs/troubleshooting.md](docs/troubleshooting.md). Only PDF export needs it.

## Running the tests

```bash
python -m pytest                    # everything
python -m pytest -m "not data"      # fast: skips the image-corpus tests
```

**Read the skip count, not just the exit code.** The suite is marked, and a marker that
skips wholesale takes a large amount of coverage with it quietly:

| Marker | Skips when | What you lose |
|---|---|---|
| `data` | `tests/test_data/` is missing (e.g. a shallow clone) | almost the whole calibration path |
| `gui` | PySide6 is not installed | every GUI test |
| `needs_opengl` | no OpenGL context | the rendering tests |
| `needs_jit` | `NUMBA_DISABLE_JIT` is set | the compiled kernels |

Do not set `NUMBA_DISABLE_JIT` to make coverage look better: it switches off the tests
that guard the compiled Jacobian, which is where the subtle bugs live. Coverage reads
low by design — the kernels are covered behaviourally by the `data` tests.

Two more things worth knowing before you file a bug against your own run:

- **OpenCV versions are not numerically identical.** Before 4.14 the `CharucoDetector`
  reports ChArUco corners half a pixel further on than 4.14 and 5.x do; pyCamSet moves
  them into the same pixel-centre frame, but sub-pixel positions still differ by a few
  hundredths of a pixel. Don't expect reprojection errors to match exactly across versions.
- **On Windows, some tests need symlink permission.** A fixture symlinks the test
  corpus; without Developer Mode enabled you get `OSError: [WinError 1314]`. That is
  your machine, not the change you are testing.

## Making a change

**One pull request, one theme.** One problem, one observable change, one diff a
reviewer can hold in their head. "Fix the three bugs I found" is three PRs. Keep
formatting sweeps and renames in their own obviously-mechanical PR so they cannot hide
a behaviour change.

Branch off `development`, which is where work lands; `master` is the release branch.

**Tests ship with the behaviour they cover.** A fix without a test that fails before it
is not finished.

**Commit messages** are a prose imperative sentence saying what the commit makes true —
*"Measure coverage on the code that ships"*, *"Make a tag prove itself before it
publishes"*. No `feat:`/`fix:` prefixes, no scope parentheses, no ticket numbers. Say
why, not just what; the diff already says what.

**Style** follows the file you are editing. There is no linter or formatter configured,
so match the surrounding code rather than introducing a house style. UK English in
comments and docs. Docstrings use reST fields (`:param x:`) and are rendered into the
API pages, so a malformed one breaks the docs build.

**Education mode.** A change that alters how calibration processes data, the options it
takes, or what a phase shows also changes what the education windows must teach. Update
that teaching in the same pull request. The rules, the table of where each kind of change
is taught, and the checks are in [The education mode](#the-education-mode).

## The education mode

Education mode is the learner-facing part of the GUI: one window per calibration phase,
in `pyCamSet/gui/education_phase1.py` to `education_phase4.py`, showing what the
calibration did to *this* run's photos and data. Its data layer is
`pyCamSet/workflow/education_*.py`. That layer reads the saved run and, where a step was
not saved, recomputes it from the photos (Phase 1 re-runs detection on one image). The
teaching text is the copy in the `pyCamSet/gui/education_*.py` modules.

**The rule.** A change that alters how calibration processes data, the options it takes,
or what a phase shows must update the education teaching in the same pull request. The
pull request must say which teaching changed and how it was checked. If you believe
nothing needs to change, say why in one line. Nothing checks this automatically: reviewers
enforce it against the table below.

| If you add or change | Teach it in | Prove it with |
|---|---|---|
| A lens or camera model (a `Camera` subclass, a `MODEL_BLOCKS` entry) | The Phase 2 lens summary and board fit: `workflow/education_visual_p2.py` (`is_telecentric`, `lens_summary`, `fit_views`, `face_warp`, `display_depths`). The drawn rig: `workflow/education_visual_p3.py`. The rig's pose and point classes: `workflow/education_derive.py` (`_phase4_point_classes`). | `tests/test_education_phase2_visual.py`, `tests/test_education_phase3.py`, `tests/test_education_derive_p4.py`. Add a `lens_model` entry to `pyCamSet/workflow/education_coverage.py`. |
| A calibration target (a `TARGET_CLASSES` line, or a new class) | Its Phase 1 detection explanation, dispatched on the explain events the target's detector emits, in `workflow/education_detect_explain.py` (`explain_detection`). A target that emits no events is shown with only the input and result steps, so its entry is `not_yet_taught` until it has a path. | `tests/test_education_detect_explain.py` (add a case to `CASES`), the row in `tools/edu_release/release_matrix.py`, and a `target` entry. |
| A face or board layout (faces, corners, spacing, a new face warp) | Phase 2 board views and face warp: `workflow/education_visual_p2.py` (`board_views`, `face_warp`). Object scale in Phase 1: `workflow/education_visual_p1.py` (`um_per_px`). | `tests/test_education_phase2_visual.py`, and the detection test above. |
| A marker dictionary or marker family | The explanation takes the dictionary from the target: aruco1 from the board's `getDictionary()`, aruco2 from `_aruco_dict_int`, both in `workflow/education_detect_explain.py`. Any wording that names a dictionary must match. | `tests/test_education_detect_explain.py` and the release matrix rows. |
| A detector, a detector backend or a detector option (`DETECTOR_BACKENDS`, `marker_backend`, a new `explain` event) | Emit the step data from the detector under the default-off `explain` hook, as `pyCamSet/calibration_targets/charuco.py`, `charuco2.py` and `puzzleboard.py` do. Read or reproduce it in `workflow/education_detect_explain.py`. Stage copy: `gui/education_phase1.py` (`_show_find`). | `tests/test_education_detect_explain.py`, including `test_explain_hooks_do_not_change_the_detection` and the consistency check against the saved detections. A `detector_backend` entry. |
| A solver, a loss or a robust-loss function (`optimisation/robust_loss.py`, the solver choice) | Phase 3 objective wording: `gui/education_phase3.py` (`_objective_header`). Phase 4 cost labels: `gui/education_phase4.py` (`cost_column_title`, `cost_axis_label`). Recorded solver steps: `workflow/education_playback.py` (`AcceptedStepCollector`, `read_accepted_step_trace`). Loss settings from saved options: `workflow/education_trace.py` (`loss_settings_from_options`). | `tests/test_education_phase4.py`, `tests/test_education_phase4_visual.py`, `tests/test_education_playback.py`. A `solver` or `loss` entry. A loss that only the generic wording covers is `partial` until a test names it. |
| Parameter handling, the gauge, a fixed pose, or a default | Re-derivation of the run's own seed: `workflow/education_derive.py` (`_seed_from_inputs` reads the saved option, for example `recover_single_face_telecentric`). The stated default at the level it applies: `gui/education_levels.py`. | `tests/test_education_derive_p3.py`, `tests/test_education_derive_p4.py`, `tests/test_education_phase3.py`. |
| A new saved-file field or diagnostic key | The education reader that needs it, with a default for runs saved without the key, so an older run still opens. Choose that default from what the run's own code did before the field existed, not from the solver's current default. `recover_single_face_telecentric` in `education_derive.py` is an example: a missing key reads as `False`, because single-face recovery did not exist for runs saved before the option was recorded, while the solver now defaults to `True`. The comment in that file gives the dates and the one-day window where the older runs cannot be told apart. | `tests/test_education_release_negatives.py` (legacy and corrupted runs) and `tests/test_education_trace.py`. |
| A new phase option, or a changed default | The option's wording at the level it applies: `gui/education_levels.py`, `gui/education_phase1.py` to `education_phase4.py`. The default written in the text must be the default the code uses. | The phase's education test file, and `tests/test_education_copy_lint.py`. |
| GUI wording or a visual change | The learner copy in `gui/education_*.py`, and the styling in `gui/education_style.py`. | `tests/test_education_copy_lint.py`, `tests/test_education_foundation.py`, and a screenshot run (below). |

The registered items, and how education mode covers each, are listed in
`pyCamSet/workflow/education_coverage.py`. `tests/test_education_coverage_contract.py` fails
when a registered target, lens model, detector backend, solver or loss has no entry, or
when an entry names code or a registry name that no longer exists. A `not_yet_taught`
entry is an honest gap rather than a pass, so it gives its reason.

**Commands.** Run them from the checkout root. The test suite sets
`QT_QPA_PLATFORM=offscreen` itself. Set it yourself when you run GUI code outside pytest.

```bash
python -m pytest -k education -q                              # the education tests
python tools/edu_release/run_tests.py --tests test_education_coverage_contract,test_education_copy_lint -- -q
python tools/edu_release/release_matrix.py --phases 1234      # 12 rows x Phases 1-4 (slow)
python tools/edu_release/release_matrix.py --table            # runs every cell (as slow as --phases 1234), prints the rows for docs/how-to/gui.md
python tools/edu_release/check_gui_doc.py --no-matrix         # the GUI support statement
python tools/edu_release/edu_screenshots.py --run <phaseN_runs/ID> --dataset <name> \
    --out <scratch folder> --size-spec 1400x900=Light --pyvista   # real dialog, one saved run
python tools/edu_release/foundation_gallery.py --out <scratch folder> --no-pyvista   # synthetic data
```

**Interpreters.** `run_tests.py` and `release_matrix.py` run the interpreter executing them by default
(`sys.executable`). `run_tests.py` also accepts `--env cv4` or `--env lean`, which read
the interpreter path from `PYCAMSET_EDU_ENV_CV4` (an OpenCV 4.11 environment) or
`PYCAMSET_EDU_ENV_LEAN` (a PySide6-free environment) and stop with a message when the
variable is not set. `run_tests.py` and `release_matrix.py` accept `--python PATH` for any
other interpreter with pyCamSet's dependencies installed; such a run is labelled `custom`
in its output. Say in the pull request which interpreter, OpenCV and OS a result came from,
and which checks did not run. Set `EDU_SHORT_TMP` to a short folder if numba's cache path
exceeds the Windows path limit.
`edu_screenshots.py` and `foundation_gallery.py` write their output under `--out`
(`foundation_gallery.py` defaults to a folder in the system temp directory), so
point it outside the repository. The screenshot tool reads a saved run and does not write
into it.

**Checks that must pass.**

- `tests/test_education_copy_lint.py` checks three things. (1) The banned-phrase list in
  `tests/test_data/education/banned_phrases.json` runs on sample strings, and also on the
  real caption, chips and Numbers text of every stage of Phases 2, 3 and 4 at all three
  levels. (2) The internal-code rule (plan codes such as `C1`, `L3-only`, `writer`) applies
  at every level, and the L1 jargon rule (`gauge`, `seed`, `D3.5`, `fixed_pose`,
  `detector-frame`) applies to the real L1 text. (3) Every quoted control label
  (‘…’) matches the GUI's own text. The banned-phrase list is pinned to the reviewed copy
  `A1-copy.md` by SHA-256. That copy is not in this repository, so the pin cannot be
  checked here. It does not cover tooltips, panel titles, or alt text.
- `tests/test_education_coverage_contract.py`: every registered target, lens model,
  detector backend, solver and loss has a coverage entry, and every entry names code that
  still exists. It skips when an optional target module is missing (the `puzzle` extra),
  and a skip is not a pass.
- The release matrix (`release_matrix.py`). Every cell is a pass with a coverage kind, or
  recorded as `untested: <reason>`. A skip never counts as a pass. The pass counts are
  minimums: at least 8 of the 12 Phase 1 cells and 22 of the 36 Phase 2 to 4 cells
  (`MIN_PASSED`). A skip must carry its reason.
- The held-run and negative tests, `tests/test_education_release_negatives.py`. Legacy,
  corrupted, mismatched and promoted runs must fail closed or be labelled, never shown as a
  normal pass.
- `check_gui_doc.py` prints `RESULT: PASS`, if you touched the support statement in
  `docs/how-to/gui.md` or the matrix it reproduces.

**When a check fails, look here.**

- *The coverage contract* (`test_education_coverage_contract.py`). Each failure names the
  item and the kind (target, lens model, detector backend, solver or loss). The entry is
  in `pyCamSet/workflow/education_coverage.py`: add it, or correct its name or its code
  reference. A `taught` entry's proof is `tests/<file>.py::<test>`, and the test must name
  the item in its source or its parametrisation id. A `partial` or `not_yet_taught` entry
  needs a reason. Do not weaken the test to pass.
- *The copy lint* (`test_education_copy_lint.py`). Each failure gives the file, the key
  (`L<level> <caption | chip | numbers row>`), the stage, and the matched text with its
  context. Reword the copy in the `gui/education_*.py` file it names. Do not edit
  `banned_phrases.json` or its pinned SHA to make it pass.
- *The release matrix*. The output lists each failing or unexplained cell as
  `<Row>-p<N>`, from `tests/test_education_release_matrix_p1.py` or
  `tests/test_education_release_matrix_p234.py`.

**Accuracy rules for taught text.**

1. Every number in a caption, chip or table is computed from the run's own saved data, at
   display time. Never hard-code a count, a figure, a camera number or a result.
2. Hedge only what the run does not establish. A verified fact is stated plainly.
3. A run that is not a usable calibration is never presented as a success. A held,
   promoted, legacy, corrupted or failed run is labelled as such, and it gets no pass chip.
4. Teaching text follows the library's actual behaviour. Read the code path, including the
   defaults in `optimisation/optimisation_handling.py` and the GUI, rather than the paper or
   memory. If the code and the text disagree, the code is right and the text changes.
5. Learner copy avoids the banned phrases, and a camera keeps its colour in every figure.

**Agents.** An agent must read this section before changing anything under `pyCamSet/` that
the table covers. It must not hard-code a figure that a learner sees. It must update
`pyCamSet/workflow/education_coverage.py` in the same change, and run
`python -m pytest -k education -q` and the coverage contract before it reports the work
done. It starts from the table row for the change. It must not weaken
`banned_phrases.json`, the `MIN_PASSED` thresholds in `release_matrix.py`, a
`not_yet_taught` reason, or the copy-lint pin. If the matrix cannot run on its interpreter,
it records the matrix as untested. In its report it must say which checks did not run.

## Don't let a test assert your machine

The skip table above is about a run being green when it should not be. This is the
opposite, and it is the one that gets caught by CI rather than by you: a test that
passes locally for a reason that is not the code.

If an assertion depends on a native library, `sys.prefix`, `PATH`, a GPU, a drive
letter, or an optional package you happen to have installed, it is asserting your
machine. It will pass every time you run it and fail on a runner that is set up
differently. Assert the mechanism instead — that the thing was attempted, that it did
the right thing *given what was available* — and make an expectation conditional on
whatever it depends on:

```python
if os.path.isdir(expected):
    assert registered == [expected]
else:
    assert registered == []        # nothing there to register
```

Be most suspicious of a test you wrote precisely because a code path is hard to reach
locally. It exists because your machine is not representative — so don't then assert
your machine's answer. If the honest answer really is environment-dependent, say which
environment you couldn't test in the PR body. Nobody minds that; a red CI job costs a
review cycle.

## What CI will run

Every pull request runs three workflows. You can run all of them locally:

```bash
python -m pytest --timeout=1800           # test matrix, 9 jobs upstream
python -m mkdocs build --strict           # docs, if you touched docs or a docstring
python -m build && twine check --strict dist/*   # packaging, if you touched pyproject
```

`mkdocs build --strict` turns warnings into failures, and it renders docstrings, so an
API docstring change can fail the docs job while every test passes. If you changed
dependencies, regenerate the pinned file and commit the result:

```bash
python setup_scripts/write_core_requirements.py
git diff --exit-code requirements_core.txt
```

Your local run covers one cell of a nine-cell matrix (three platforms × Python
versions, plus OpenCV `<5` and lean-install jobs). Upstream CI covers the rest.

## Opening the pull request

Say what the user-visible problem was before the implementation detail. Include:

- the commands you ran, their real results, and **the checks that did not run**;
- if the change alters calibration processing or a phase, which education teaching you
  changed, or why none needed to change ([The education mode](#the-education-mode));
- known limitations, and any question you want the maintainer to answer;
- anything you could not verify on your machine, said plainly.

Be detailed and be brief — those are not in tension. Give the numbers rather than
adjectives, say each thing once, and leave out how you arrived at the change. But never
shorten by dropping a caveat: an unverified branch or an open question is the last
thing to cut, not the first.

If your change is AI-assisted, that is fine; review it yourself first, and do not
present unverified output as tested.

## Reporting a bug

Include the output of:

```bash
python -c "import sys, numpy, cv2, numba; print(sys.version, numpy.__version__, cv2.__version__, numba.__version__)"
```

plus your OS, how you installed pyCamSet, and the smallest script that reproduces it. A
calibration issue is much easier to act on with the target type and a description of
the rig.

## Licence

pyCamSet is Apache-2.0, and contributions are accepted under the same licence. If you
bring in third-party code, say where it came from and under what terms.
