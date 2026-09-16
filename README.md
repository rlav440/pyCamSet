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


###  Installing via PyPI

Here is the [link](https://pypi.org/project/pyCamSet/) to our project on PyPI

```
pip install pyCamSet
```

This is the full install: the library, plotting, and the graphical
calibration workflow, which is launched with

```
pycamset
```

#### The headless install

PySide6 is a large dependency, so PyCamSet will also operate headlessly.
Install via:

```
pip install pyCamSet --no-deps
pip install -r requirements_core.txt
```

Everything except the GUI works from that install, plotting included;
`pycamset` reports what to install if it is run. `requirements_core.txt`
is generated from `pyproject.toml` by
`setup_scripts/write_core_requirements.py`.

#### Optional extras

Open3D powers the lockbox editor's viewport and one visualisation path;
everything it draws has a PyVista equivalent, and PyVista is included by
default, so it is kept separate:

```
pip install "pyCamSet[viz]"
```

Optuna, for the detection optimisation study workflow:

```
pip install "pyCamSet[optimisation]"
```

The PuzzleBoard calibration target is provided by [PuzzleBoard repository](https://github.com/PStelldinger/PuzzleBoard) by Peer Stelldinger and the HAW Hamburg authors. 
To use the PuzzleBoard target, install the optional dependency:

```
pip install "puzzle_board @ git+https://github.com/PStelldinger/PuzzleBoard.git"
```

### Troubleshooting

Native Cairo setup, the optional PuzzleBoard dependency, the
PuzzleBoardCube face-size limit and the `'NoneType' object is not callable`
error you get when a target's graphics dependencies are missing are all
covered in the
[troubleshooting guide](https://rlav440.github.io/pyCamSet/dev/troubleshooting/).

## Reporting issues
To report an issue or suggest a new feature, please use the [issues page](https://github.com/rlav440/pyCamSet/issues).
Please check existing issues before submitting a new one.

## Contributors

Robin
Collin 
Raymon

## Acknowledgements
