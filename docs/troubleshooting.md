# Troubleshooting

## A target class is `None`

`ChArUco`, `Ccube`, `ChArUco2`, `Ccube2`, `PuzzleBoard`, `PuzzleBoardCube`,
`CIco`, `CIco2` and `PuzzleBoardIco` are imported behind a guard, because they need graphics libraries that a
reconstruction-only install does not. When one of those imports fails, the name
is bound to `None` rather than raising, so the failure surfaces later and
somewhere else:

```text
TypeError: 'NoneType' object is not callable
```

To see the real cause, import the target's module directly:

```python
import pyCamSet.calibration_targets.charuco
```

That raises the underlying `ImportError` — most often a missing `svgwrite` or
`cairosvg`, or the native Cairo library those depend on.

## The native Cairo library

`cairosvg` depends on `cairocffi`, which needs the native `cairo` library
installed separately; `pip` alone cannot provide this reliably on Windows.
Without it, writing a target to PDF fails with an `OSError` about a missing
`cairo-2` library. Only the PDF formats are affected: `cairosvg` is imported
where it is used, so importing a target module, or writing one to SVG, works
without it.

pyCamSet already tries to fix the common case for you: every place that
imports `cairosvg` first imports
[`ensure_cairo_dll_available`][pyCamSet.utils.cairo_dll_helper.ensure_cairo_dll_available]'s
module, which locates the conda environment's library directory and registers
it. It is a no-op where Cairo already loads.

On Windows only, if `cairosvg` still cannot find Cairo at the moment a target is
drawn (a plain virtualenv built on a conda interpreter, or a headless process
that never inherited an activated environment on its `PATH`), a wider search
runs once, inside `cairosvg_or_explain`, not at import time. It tries each
conda environment and installation the process can see, but only a
`Library\bin` that holds a DLL under a name `cairocffi` loads (`cairo-2.dll`,
`cairo.dll` or `libcairo-2.dll`), so `cairo-gobject.dll` alone does not
qualify. A directory that does not make the import work is unregistered again,
and `PATH` is left as it was. A machine with no Cairo at all is still reported
honestly, and the tests that need it skip rather than fail.

When that is not enough, install the native library for your platform:

| Platform | Command |
|---|---|
| conda (any OS) | `conda install -c conda-forge cairo` |
| macOS (Homebrew) | `brew install cairo` |
| Debian / Ubuntu | `apt install libcairo2` |

!!! note

    The Debian package name is a best guess and has not been verified by the
    pyCamSet team; it may differ on other distributions.

In a conda environment the whole graphics stack can go in at once:

```bash
conda install -c conda-forge cairo cairosvg svgwrite
```

Afterwards `import pyCamSet.calibration_targets.charuco` should succeed.

## `pycamset` will not start

The GUI needs PySide6, which the lean install leaves out. The command says so
rather than failing obscurely:

```text
The pyCamSet GUI could not start: <the import error>.
This is the lean install, which leaves the GUI toolkit out.
Add it with:
    pip install PySide6
```

Everything except the GUI works from a lean install, plotting included — and
the whole calibration is reachable from a script through
[`calibrate_cameras`](how-to/calibrate.md), with no GUI toolkit involved.

## PuzzleBoard is not installed

The PuzzleBoard targets depend on the upstream
[PuzzleBoard repository](https://github.com/PStelldinger/PuzzleBoard) by Peer
Stelldinger and the HAW Hamburg authors. It is a research codebase that is not
published on PyPI, so it is optional. It is needed to detect, not to make:
constructing, plotting and printing a PuzzleBoard target work without it,
because the package is imported only when an image is detected. Without it
the GUI refuses to start detection, calibration or optimisation with a
message naming the missing package, and a script that detects anyway
(through `calibrate_cameras` or `find_in_image`) fails with
`ModuleNotFoundError: No module named 'puzzle_board'`. The fix is:

```bash
pip install --no-deps "puzzle_board @ git+https://github.com/PStelldinger/PuzzleBoard.git"
```

The PuzzleBoard source is released under CC0 upstream. Retain the upstream
attribution, and cite the original PuzzleBoard work when publishing results
that use this target.

## aruco2 is not installed

ChArUco2, Ccube2 and CIco2 are printed with aruco2 markers, and the classic
targets read with `marker_backend="aruco2"` use its detector. Without it,
building one of those targets raises an `ImportError` that names this fix.
aruco2 is not on PyPI, so pip builds it from the `third_party/aruco2` submodule
of a pyCamSet checkout. The install scripts do this for you:

```bash
setup_scripts/install_optional_linux.sh
setup_scripts/install_optional_macos.sh
```

```powershell
powershell -ExecutionPolicy Bypass -File setup_scripts\install_optional_windows.ps1
```

The build needs two things pip cannot install. The scripts check for both:

| | Windows | macOS | Linux |
|---|---|---|---|
| C++ compiler | Visual Studio Build Tools, "Desktop development with C++" | `xcode-select --install` | `build-essential` or equivalent |
| OpenCV 4.9+ with its CMake files | the release from [opencv.org/releases](https://opencv.org/releases/), with `OPENCV_DIR` set to its `build` folder | `brew install opencv` (found automatically) | conda-forge `libopencv`, with `OPENCV_DIR` set to its `lib/cmake/opencv4` |

OpenCV 4.9 is the oldest release whose headers aruco2 compiles against. Ubuntu
24.04's `libopencv-dev` is 4.6 and too old. The built module links to this
native OpenCV, not to the `opencv-contrib-python` wheel pyCamSet imports as
`cv2`; the two coexist in one process. On Windows the build copies OpenCV's
`opencv_world` DLL into the installed package, so nothing has to go on `PATH`.

To build by hand instead, run `git submodule update --init third_party/aruco2`,
then `pip install -v ./third_party/aruco2`, adding
`--config-settings=cmake.define.OpenCV_DIR=<that folder>` for an OpenCV
CMake does not find by itself. On Windows, write the folder with forward
slashes, and when a Visual Studio newer than the OpenCV release cannot find
its binaries, add `--config-settings=cmake.define.OpenCV_ARCH=x64` and
`--config-settings=cmake.define.OpenCV_RUNTIME=vc16` (the `x64/vcNN` folder the
release ships). `CITATION.md` says which builds aruco2's own release check covers.

## The MATLAB stereo matcher

`reconstruction_utils.stereo_reconstruct(..., matlab=True)` computes the
disparity with MATLAB's `disparitySGM` instead of OpenCV's StereoSGBM, the
default. It needs a licensed MATLAB with the Computer Vision Toolbox and the
MATLAB Engine API for Python (`matlab.engine`) installed into the environment.
The install scripts leave it out, and nothing else in pyCamSet uses it.

## PuzzleBoardCube rejects `n_points`

`PuzzleBoardCube` is a modified implementation of the original PuzzleBoard
target, which is a single planar board; the cube version is intended to work
like the Ccube. It assigns six disjoint, deterministic windows of one periodic
PuzzleBoard code to the cube faces, in the fixed order front, right, back,
left, top, bottom. Its `puzzle-cube-v1` layout is not random: identical
`n_points` and `length` produce identical face patterns and cube geometry.

The maximum face size is **167 squares per side**, set by packing six faces into
the 501-position code field three across and two down. Larger values are
rejected rather than being allowed to overlap or exceed the code period:

```text
ValueError: n_points must not exceed 167 for puzzle-cube-v1
```

See [PuzzleBoardCube](targets/puzzleboard-cube.md).

## OpenCV version differences

pyCamSet supports OpenCV 4.8 and above, including 5.x, and handles the API
differences at the call sites.

**Every detector returns points in the pixel-centre frame, whichever OpenCV
is installed.** Integer coordinates are pixel centres, the frame
`cv2.calibrateCamera` and `cv2.projectPoints` assume. The `CharucoDetector` of
OpenCV 4.14 and later returns ChArUco corners in it; earlier versions return them
half a pixel further on, on both axes (measured: +0.5 px on 4.11 and 4.13, 0 on
4.14 and 5.0), and `aruco_opencv.charuco_corners_to_pixel_centre` moves them back,
so ChArUco, Ccube and CIco give the same points under any version and the same
frame as ChArUco2, Ccube2, CIco2 and PuzzleBoard. Sub-pixel positions still differ
by a few hundredths of a pixel between versions. A detection cache written before
this frame was recorded is detected again; a camset saved before it may have
its principal points half a pixel on from a newer one's, so re-run its
calibration before comparing the two.

pyCamSet depends on `opencv-contrib-python`. Every `opencv-*` wheel installs
the same `cv2` package, so an environment that already has `opencv-python`
holds two copies of one package, and uninstalling either breaks `import cv2`.
pyCamSet warns on import when it finds more than one. Keep one, reinstalling
the OpenCV major version you were on:

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless
pip install --force-reinstall --no-deps "opencv-contrib-python>=4.8,<5"   # or ">=5"
```
