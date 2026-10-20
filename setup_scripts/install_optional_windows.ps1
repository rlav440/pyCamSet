# Install pyCamSet's optional dependencies with pip, on Windows.
#
#   powershell -ExecutionPolicy Bypass -File setup_scripts\install_optional_windows.ps1 [-SkipAruco2]
#
# Run it from a checkout of the repository, in the environment pyCamSet is
# installed in. It installs:
#
#   - the [optimisation] and [viz] extras (Optuna, Open3D), read from pyproject.toml;
#   - PuzzleBoard, for the PuzzleBoard targets (not on PyPI: installed from its git repository);
#   - aruco2, for ChArUco2, Ccube2, CIco2 and marker_backend="aruco2" (not on PyPI: pip builds
#     it from the third_party/aruco2 submodule).
#
# pip fetches everything aruco2's build needs except two things it cannot supply: the MSVC C++
# compiler (Visual Studio Build Tools, "Desktop development with C++") and an OpenCV of
# version 4.9 or newer with its CMake files, such as the Windows release from
# https://opencv.org/releases/ unpacked anywhere. Point the script at that OpenCV's build
# folder, the one holding OpenCVConfig.cmake:
#   $env:OPENCV_DIR = "C:\opencv\build"
# The build copies OpenCV's opencv_world DLL into the installed aruco2 package, so nothing has
# to be added to PATH afterwards. The script checks for both prerequisites and, when one is
# missing, skips aruco2 and says what to install.
#
# Environment: PYTHON (default python), OPENCV_DIR (see above).
# Exits non-zero if anything it tried to install failed or was skipped.
# Written for Windows PowerShell 5.1 as well as PowerShell 7.
param([switch]$SkipAruco2)

$ErrorActionPreference = "Continue"
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Root = Split-Path -Parent $PSScriptRoot
$failed = New-Object System.Collections.Generic.List[string]

function Step([string]$Text) { Write-Host ""; Write-Host "== $Text" }

Step "Python"
& $Python -c "import sys; sys.exit(sys.version_info < (3, 11))" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "pyCamSet needs Python 3.11 or newer; '$Python' is not it. Set `$env:PYTHON to its path."
    exit 1
}
& $Python --version

Step "Optuna and Open3D (the [optimisation] and [viz] extras)"
# Read the package lists from pyproject.toml, so this script cannot drift from the extras.
# Single quotes only: Windows PowerShell 5.1 drops double quotes from a native command's arguments.
$readExtras = 'import sys, tomllib; g = tomllib.load(open(sys.argv[1], ''rb''))[''project''][''optional-dependencies'']; print(chr(10).join(g[''optimisation''] + g[''viz'']))'
$extras = @(& $Python -c $readExtras (Join-Path $Root "pyproject.toml"))
if ($LASTEXITCODE -ne 0 -or $extras.Count -eq 0) {
    Write-Error "Could not read $(Join-Path $Root 'pyproject.toml')"
    exit 1
}
& $Python -m pip install @extras
if ($LASTEXITCODE -ne 0) { $failed.Add("extras: $($extras -join ' ')") }

Step "PuzzleBoard"
# --no-deps, as the install docs and CI install it: pyCamSet's own dependencies cover it.
& $Python -m pip install --no-deps "puzzle_board @ git+https://github.com/PStelldinger/PuzzleBoard.git"
if ($LASTEXITCODE -ne 0) { $failed.Add("puzzle_board (needs git on PATH)") }

Step "aruco2"
$missing = New-Object System.Collections.Generic.List[string]
if ($SkipAruco2) {
    $failed.Add("aruco2 (skipped: -SkipAruco2)")
} else {
    $aruco2 = Join-Path $Root "third_party\aruco2"
    if (-not (Test-Path (Join-Path $aruco2 "pyproject.toml"))) {
        git -C $Root submodule update --init third_party/aruco2
        if ($LASTEXITCODE -ne 0) {
            $missing.Add("the third_party/aruco2 submodule (git submodule update --init third_party/aruco2)")
        }
    }

    # vswhere ships with every Visual Studio and Build Tools install since 2017.
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $msvc = $null
    if (Test-Path $vswhere) {
        $msvc = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    }
    if (-not $msvc) {
        $missing.Add("the MSVC C++ compiler (Visual Studio Build Tools with 'Desktop development with C++': https://visualstudio.microsoft.com/visual-cpp-build-tools/)")
    }

    $configSettings = @()
    if (-not $env:OPENCV_DIR) {
        $missing.Add("an OpenCV 4.9+ with its CMake files (download the Windows release from https://opencv.org/releases/, unpack it, and set `$env:OPENCV_DIR to its build folder)")
    } elseif (-not (Test-Path (Join-Path $env:OPENCV_DIR "OpenCVConfig.cmake"))) {
        $missing.Add("OpenCVConfig.cmake in OPENCV_DIR=$($env:OPENCV_DIR) (point it at the release's build folder)")
    } else {
        # OpenCVConfig-version.cmake states the version CMake will find.
        $versionFile = Join-Path $env:OPENCV_DIR "OpenCVConfig-version.cmake"
        $match = Select-String -Path $versionFile -Pattern '^set\(OpenCV_VERSION ([0-9.]+)\)' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($match) {
            $version = [version]($match.Matches[0].Groups[1].Value)
            if ($version -lt [version]"4.9") {
                $missing.Add("OpenCV 4.9 or newer (OPENCV_DIR has $version)")
            }
        }
        Write-Host "Building against the OpenCV in $($env:OPENCV_DIR)"
        # Forward slashes: aruco2's CMakeLists builds the DLL paths it installs from OpenCV_DIR,
        # and CMake reads a backslash in them as an escape.
        $cmakeOpenCVDir = $env:OPENCV_DIR -replace '\\', '/'
        $configSettings = @("--config-settings=cmake.define.OpenCV_DIR=$cmakeOpenCVDir")
        # The release pack's OpenCVConfig.cmake picks its x64\vcNN folder from the compiler
        # version and knows only the compilers that existed when it was released, so a newer
        # Visual Studio finds "no binaries compatible with your configuration". MSVC is
        # binary-compatible from VS 2015 (vc14) on, so name the folder the pack ships.
        $runtime = Get-ChildItem -Path (Join-Path $env:OPENCV_DIR "x64") -Directory -Filter "vc*" -ErrorAction SilentlyContinue |
            Sort-Object { [int]($_.Name.Substring(2)) } | Select-Object -Last 1
        if ($runtime) {
            $configSettings += "--config-settings=cmake.define.OpenCV_ARCH=x64"
            $configSettings += "--config-settings=cmake.define.OpenCV_RUNTIME=$($runtime.Name)"
        }
    }

    if ($missing.Count -eq 0) {
        & $Python -m pip install -v $aruco2 @configSettings
        if ($LASTEXITCODE -eq 0) { & $Python -c "import aruco2" }
        if ($LASTEXITCODE -ne 0) { $failed.Add("aruco2 (build or import failed; see the output above)") }
    } else {
        Write-Host "Skipping aruco2. Missing:"
        foreach ($item in $missing) { Write-Host "  - $item" }
        $failed.Add("aruco2 (prerequisites missing, listed above)")
    }
}

Step "Summary"
if ($failed.Count -eq 0) {
    Write-Host "All optional dependencies are installed."
    exit 0
}
Write-Host "Not installed:"
foreach ($item in $failed) { Write-Host "  - $item" }
exit 1
