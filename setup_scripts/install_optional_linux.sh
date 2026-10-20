#!/usr/bin/env bash
# Install pyCamSet's optional dependencies with pip, on Linux.
#
#   setup_scripts/install_optional_linux.sh [--skip-aruco2]
#
# Run it from a checkout of the repository, in the environment pyCamSet is
# installed in. It installs:
#
#   - the [optimisation] and [viz] extras (Optuna, Open3D), read from pyproject.toml;
#   - PuzzleBoard, for the PuzzleBoard targets (not on PyPI: installed from its git repository);
#   - aruco2, for ChArUco2, Ccube2, CIco2 and marker_backend="aruco2" (not on PyPI: pip builds
#     it from the third_party/aruco2 submodule).
#
# pip fetches everything aruco2's build needs except two things it cannot supply: a C++
# compiler, and an OpenCV development package of version 4.9 or newer (Ubuntu 24.04's
# libopencv-dev is 4.6 and too old). The script checks for both and, when one is missing,
# skips aruco2 and says what to install. Point it at an OpenCV with
#   OPENCV_DIR=/path/to/lib/cmake/opencv4
#
# Environment: PYTHON (default python3), OPENCV_DIR (optional, see above).
# Exits non-zero if anything it tried to install failed or was skipped.
set -u

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIP_ARUCO2=0
[ "${1:-}" = "--skip-aruco2" ] && SKIP_ARUCO2=1

failed=()
step() { printf '\n== %s\n' "$*"; }

step "Python"
if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    echo "pyCamSet needs Python 3.11 or newer; '$PYTHON' is not it. Set PYTHON=/path/to/python." >&2
    exit 1
fi
"$PYTHON" --version

step "Optuna and Open3D (the [optimisation] and [viz] extras)"
# Read the package lists from pyproject.toml, so this script cannot drift from the extras.
extras="$("$PYTHON" -c '
import sys, tomllib
with open(sys.argv[1], "rb") as fh:
    groups = tomllib.load(fh)["project"]["optional-dependencies"]
print(" ".join(groups["optimisation"] + groups["viz"]))
' "$ROOT/pyproject.toml")" || { echo "Could not read $ROOT/pyproject.toml" >&2; exit 1; }
# Unquoted on purpose: one argument per package.
"$PYTHON" -m pip install $extras || failed+=("extras: $extras")

step "PuzzleBoard"
# --no-deps, as the install docs and CI install it: pyCamSet's own dependencies cover it.
"$PYTHON" -m pip install --no-deps "puzzle_board @ git+https://github.com/PStelldinger/PuzzleBoard.git" \
    || failed+=("puzzle_board (needs git on PATH)")

step "aruco2"
aruco2_missing=()
if [ "$SKIP_ARUCO2" = 1 ]; then
    failed+=("aruco2 (skipped: --skip-aruco2)")
else
    if [ ! -f "$ROOT/third_party/aruco2/pyproject.toml" ]; then
        git -C "$ROOT" submodule update --init third_party/aruco2 \
            || aruco2_missing+=("the third_party/aruco2 submodule (git submodule update --init third_party/aruco2)")
    fi
    command -v c++ >/dev/null 2>&1 \
        || aruco2_missing+=("a C++ compiler (Debian/Ubuntu: apt install build-essential)")
    cmake_args=()
    if [ -n "${OPENCV_DIR:-}" ]; then
        if [ ! -f "$OPENCV_DIR/OpenCVConfig.cmake" ]; then
            aruco2_missing+=("OpenCVConfig.cmake in OPENCV_DIR=$OPENCV_DIR")
        else
            # OpenCVConfig-version.cmake states the version CMake will find.
            version="$(sed -n 's/^set(OpenCV_VERSION \([0-9.]*\)).*/\1/p' "$OPENCV_DIR/OpenCVConfig-version.cmake" 2>/dev/null)"
            if [ -n "$version" ] && ! "$PYTHON" -c "import sys; v = tuple(int(p) for p in '$version'.split('.')[:2]); sys.exit(v < (4, 9))"; then
                aruco2_missing+=("OpenCV 4.9 or newer (OPENCV_DIR has $version)")
            fi
        fi
        # Link to that OpenCV by the path it is at now, as the docs build does.
        libdir="$(cd "$OPENCV_DIR/../.." 2>/dev/null && pwd)"
        cmake_args=(--config-settings=cmake.define.OpenCV_DIR="$OPENCV_DIR"
                    --config-settings=cmake.define.CMAKE_INSTALL_RPATH="$libdir"
                    --config-settings=cmake.define.CMAKE_BUILD_WITH_INSTALL_RPATH=ON)
    elif version="$(pkg-config --modversion opencv4 2>/dev/null)"; then
        if ! "$PYTHON" -c "import sys; v = tuple(int(p) for p in '$version'.split('.')[:2]); sys.exit(v < (4, 9))"; then
            aruco2_missing+=("OpenCV 4.9 or newer (found $version through pkg-config; conda-forge's libopencv is new enough: set OPENCV_DIR to its lib/cmake/opencv4)")
        fi
    else
        aruco2_missing+=("an OpenCV 4.9+ development package (none found by pkg-config; install one, e.g. conda-forge libopencv, and set OPENCV_DIR to its lib/cmake/opencv4)")
    fi
    if [ ${#aruco2_missing[@]} -eq 0 ]; then
        "$PYTHON" -m pip install -v "$ROOT/third_party/aruco2" ${cmake_args[@]+"${cmake_args[@]}"} \
            && "$PYTHON" -c "import aruco2" \
            || failed+=("aruco2 (build or import failed; see the output above)")
    else
        echo "Skipping aruco2. Missing:" >&2
        printf '  - %s\n' "${aruco2_missing[@]}" >&2
        failed+=("aruco2 (prerequisites missing, listed above)")
    fi
fi

step "Summary"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All optional dependencies are installed."
else
    echo "Not installed:"
    printf '  - %s\n' "${failed[@]}"
    exit 1
fi
