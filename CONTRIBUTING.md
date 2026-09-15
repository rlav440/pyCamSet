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

- **OpenCV 4 and 5 are not numerically comparable.** OpenCV 5 shifts detected ChArUco
  corners by about half a pixel. Don't compare reprojection errors across the majors.
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
