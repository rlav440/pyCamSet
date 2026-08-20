
# pyCamSet

pyCamSet is a python library for multi-camera calibration for MVS systems and instrumentation.

![Python 3](https://img.shields.io/badge/Python->=3.11-blue)
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GitHub issues-closed][issues-closed-shield]][issues-url]
[![License][license-shield]][license-url]
[![Contributor Covenant][code-of-conduct-shield]](CODE_OF_CONDUCT.md)
[![PyPI version fury.io][pypi-shield]][pypi-url]

[contributors-shield]: https://img.shields.io/github/contributors/rlav440/pyCamSet.svg?style=flat-square
[contributors-url]: https://github.com/rlav440/pyCamSet/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/rlav440/pyCamSet.svg?style=flat-square
[stars-url]: https://github.com/rlav440/pyCamSet/stargazers
[issues-shield]: https://img.shields.io/github/issues/rlav440/pyCamSet.svg?style=flat-square
[issues-url]: https://github.com/rlav440/pyCamSet/issues
[issues-closed-shield]: https://img.shields.io/github/issues-closed/rlav440/pyCamSet.svg
[issues-closed-url]: https://GitHub.com/SPARC-FAIR-Codeathon/sparc-me/issues?q=is%3Aissue+is%3Aclosed
[license-shield]: https://img.shields.io/github/license/rlav440/pyCamSet.svg?style=flat-square
[license-url]: https://github.com/rlav440/pyCamSet/blob/master/LICENSE
[code-of-conduct-shield]: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg
[pypi-shield]: https://badge.fury.io/py/pyCamSet.svg
[pypi-url]: https://pypi.python.org/pypi/pyCamSet/

## Table of contents
* [About](#about)
* [Getting started](#getting-started)
* [Contributing](#contributing)
* [Reporting issues](#reporting-issues)
* [Contributors](#contributors)
* [Acknowledgements](#acknowledgements)


## About

## Getting started

### Pre-requisites
- Python versions:
   - 3.11

#### Native cairo library (for ChArUco, Ccube, PuzzleBoard, and PuzzleBoardCube targets)

pyCamSet's target-generation and detection code for ChArUco, Ccube,
PuzzleBoard, and PuzzleBoardCube imports `cairosvg` at module level.
`cairosvg` depends on `cairocffi`, which needs the native `cairo` library
to be installed separately — `pip` alone cannot provide this reliably on
Windows. If the native library is missing, importing any of these target
modules will fail with an `OSError` about a missing `cairo-2` library.

**Install the native cairo library before importing these modules:**

- **conda (Windows/Linux/macOS):** `conda install -c conda-forge cairo`
- **Debian/Ubuntu:** `apt install libcairo2` *(best-guess — not verified
  by the pyCamSet team; the system package name may differ on other
  distributions)*
- **macOS (Homebrew):** `brew install cairo`

After installing the native library, `pip install cairosvg` (or
`conda install -c conda-forge cairosvg`) should work without errors, and
`import pyCamSet.calibration_targets.target_charuco` will succeed.
###  Installing via PyPI

Here is the [link](https://pypi.org/project/pyCamSet/) to our project on PyPI
```
pip install pyCamSet
```

### PuzzleBoard target

The PuzzleBoard calibration target is provided by and depends on the upstream
[PuzzleBoard repository](https://github.com/PStelldinger/PuzzleBoard) by Peer
Stelldinger and the HAW Hamburg authors. Because that repository is a research
codebase that is not published on PyPI, it is an *optional* dependency: the
core pyCamSet install does not pull it in, and the PuzzleBoard target is only
available when it is installed separately.

To use the PuzzleBoard target, install the optional dependency:

```powershell
pip install "pyCamSet[puzzle]"
```

For the vector SVG/PDF target generators, also install the graphics
dependencies in the active conda environment:

```powershell
conda activate [your calibration env here]
conda install -c conda-forge cairo cairosvg svgwrite
```

The PuzzleBoard source is released under CC0 in its upstream repository. Users
should retain the upstream attribution and cite the original PuzzleBoard work
when publishing results that use this target.

### PuzzleBoard Cube target

The `PuzzleBoardCube` target assigns six disjoint, deterministic windows of the
same periodic PuzzleBoard code to the cube faces in the fixed order front,
right, back, left, top, bottom. Its `puzzle-cube-v1` layout is non-random and
reproducible: identical face-size and square-size parameters produce identical
face patterns and cube geometry. This is a modified implementation of the original
PuzzleBoard target, which is a single planar target. The cube version is intended
to be similar to the ccube target.

The maximum face size is **167 squares per side**. This limit is enforced by
the three-column, two-row face layout, which tiles the 501x501 PuzzleBoard code
field exactly without overlap. The cube generator rejects larger values rather
than allowing windows to overlap or exceed the code period.

## Reporting issues
To report an issue or suggest a new feature, please use the [issues page](https://github.com/rlav440/pyCamSet/issues).
Please check existing issues before submitting a new one.

## Contributors

## Acknowledgements
