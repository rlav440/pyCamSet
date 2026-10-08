"""
Finding a target in an image must not import matplotlib, scipy, pyvista or numba.

Every calibration target is built from :mod:`pyCamSet.calibration_targets.core`,
and that used to import all four at module scope: ``general_utils`` for a
plot and a quaternion average, ``abstract_target`` through
``pyCamSet.cameras``, and ``shape_by_faces`` for pyvista and the numba
helpers.  A caller that only builds a board and looks for it in an image --
a live preview, say -- paid well over a second and a hundred megabytes for
none of it.  Each module now imports those where they are used.

Two things could quietly undo that or break a caller, and these tests pin
both down.  A single new module-scope import anywhere on the path brings
the lot back, so the import graph is checked in a fresh subprocess, where an
earlier test cannot have imported them already.  And a caller that reached
into one of those modules for a name it had imported, ``general_utils.plt``
say, must still find it there.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from pyCamSet.calibration_targets.core.target_registry import TARGET_NAMES
from pyCamSet.utils.lazy_imports import lazy_attributes

from conftest import REPO_ROOT, skip_without_aruco2

#: The packages none of the target path needs.  vtkmodules is pyvista's VTK
#: and llvmlite numba's compiler, either of which alone is most of the cost.
HEAVY = ("matplotlib", "scipy", "pyvista", "vtkmodules", "numba", "llvmlite")

#: The prelude every subprocess shares: a function naming the heavy
#: packages that are imported so far.
_PRELUDE = (
    "import sys\n"
    f"HEAVY = {HEAVY!r}\n"
    "def heavy():\n"
    "    return sorted({m.split('.')[0] for m in sys.modules} & set(HEAVY))\n"
)


def _run(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a fresh interpreter, from the repository root."""
    return subprocess.run(
        [sys.executable, "-c", _PRELUDE + code],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, cwd=REPO_ROOT,
    )


def _assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"Subprocess failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout, result.stdout + result.stderr


def test_importing_every_target_imports_none_of_them():
    """The package, and every module the registry names, import lean."""
    _assert_ok(_run(
        "import importlib\n"
        "import pyCamSet.calibration_targets\n"
        "from pyCamSet.calibration_targets.core import target_registry\n"
        "from pyCamSet.calibration_targets.markers import backend_registry\n"
        "for module, _ in target_registry.TARGET_CLASSES.values():\n"
        "    try:\n"
        "        importlib.import_module(module)\n"
        "    except ImportError:\n"
        "        pass  # an optional detector; nothing of it was imported\n"
        "assert not heavy(), heavy()\n"
        "print('OK')\n"
    ))


@pytest.mark.parametrize("name", TARGET_NAMES)
def test_building_a_target_imports_none_of_them(name):
    """A target built with its defaults still has not needed any of them.

    Every shipped solid is folded from face transforms already worked out,
    so building one needs neither pyvista's meshes nor numba's fitting.
    """
    skip_without_aruco2(name)
    result = _run(
        "from pyCamSet.calibration_targets.core.target_registry import build_target\n"
        "try:\n"
        f"    build_target({{'type': {name!r}}})\n"
        "except ImportError as exc:\n"
        "    print('SKIP', exc); raise SystemExit(0)\n"
        "assert not heavy(), heavy()\n"
        "print('OK')\n"
    )
    if result.returncode == 0 and result.stdout.startswith("SKIP"):
        pytest.skip(f"{name} needs an optional detector: {result.stdout[5:]}")
    _assert_ok(result)


@pytest.mark.parametrize("name", ["ChArUco", "Ccube", "ChArUco2", "Ccube2"])
def test_finding_a_target_imports_none_of_them(name):
    """Detection itself stays lean too.

    The PuzzleBoard family is left out: its third-party detector imports
    scikit-image, which uses scipy for real.
    """
    skip_without_aruco2(name)
    _assert_ok(_run(
        "import numpy as np\n"
        "from pyCamSet.calibration_targets.core.target_registry import build_target\n"
        f"target = build_target({{'type': {name!r}}})\n"
        "image = np.zeros((480, 640), np.uint8)\n"
        "target.find_in_image(image)\n"
        "assert not heavy(), heavy()\n"
        "print('OK')\n"
    ))


def test_the_names_those_modules_imported_are_still_theirs():
    """Each name a module used to import resolves to what it was."""
    import matplotlib.pyplot
    import pyvista
    from scipy.spatial.transform import Rotation

    from pyCamSet.calibration_targets import ccube, charuco, puzzleboard
    from pyCamSet.calibration_targets.core import abstract_target, shape_by_faces
    from pyCamSet.cameras import Camera, CameraSet
    from pyCamSet.cameras.lens_models import LENS_MODELS
    from pyCamSet.cameras.telecentric_camera import TelecentricCamera
    from pyCamSet.optimisation import compiled_helpers
    from pyCamSet.utils import general_utils

    assert general_utils.plt is matplotlib.pyplot
    assert general_utils.R is Rotation
    assert abstract_target.Camera is Camera
    assert abstract_target.CameraSet is CameraSet
    assert abstract_target.LENS_MODELS is LENS_MODELS
    assert abstract_target.TelecentricCamera is TelecentricCamera
    assert shape_by_faces.pv is pyvista
    assert shape_by_faces.plt is matplotlib.pyplot
    assert (shape_by_faces.n_estimate_rigid_transform
            is compiled_helpers.n_estimate_rigid_transform)
    for module in (ccube, charuco, puzzleboard):
        assert module.Camera is Camera
        assert module.plt is matplotlib.pyplot

    # The ``from module import name`` spelling goes through the same hook.
    from pyCamSet.calibration_targets.core.abstract_target import (
        Camera as imported,
    )
    assert imported is Camera


def test_lazy_attributes_resolves_only_the_names_it_was_given():
    """A name it was given is imported; any other is an AttributeError."""
    getattr_ = lazy_attributes("some.module", {
        "path": ("os.path", None),
        "join": ("os.path", "join"),
    })

    assert getattr_("path") is os.path
    assert getattr_("join") is os.path.join
    with pytest.raises(AttributeError, match="'some.module' has no attribute 'nope'"):
        getattr_("nope")
