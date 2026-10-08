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
    """The package, and every module the registry names, import lean.

    Every one of them imports without its optional detector installed, so an
    ImportError here is a broken import on the path, and fails.
    """
    _assert_ok(_run(
        "import importlib\n"
        "import pyCamSet.calibration_targets\n"
        "from pyCamSet.calibration_targets.core import target_registry\n"
        "from pyCamSet.calibration_targets.markers import backend_registry\n"
        "for module, _ in target_registry.TARGET_CLASSES.values():\n"
        "    importlib.import_module(module)\n"
        "assert not heavy(), heavy()\n"
        "print('OK')\n"
    ))


@pytest.mark.parametrize("name", TARGET_NAMES)
def test_building_a_target_imports_none_of_them(name):
    """A target built with its defaults still has not needed any of them.

    Every shipped solid is folded from face transforms already worked out,
    so building one needs neither pyvista's meshes nor numba's fitting.
    Only the targets printed with aruco2 need an optional package to be
    built, and those are skipped without it; the PuzzleBoard detector is not
    needed until detection.  Any other ImportError fails.
    """
    skip_without_aruco2(name)
    _assert_ok(_run(
        "from pyCamSet.calibration_targets.core.target_registry import build_target\n"
        f"build_target({{'type': {name!r}}})\n"
        "assert not heavy(), heavy()\n"
        "print('OK')\n"
    ))


#: How each target's own board is drawn, as an expression of ``target``: a
#: uint8 image, with a white margin, that it can be found in.
_RENDERED = {
    "ChArUco": "np.pad(target.board.generateImage((700, 700)), 50, constant_values=255)",
    "Ccube": "np.pad(np.asarray(target.textures[0], np.uint8), 50, constant_values=255)",
    "ChArUco2": "target._render(300, border_width=10.0)[0]",
    "Ccube2": "np.pad(np.asarray(target.textures[0], np.uint8), 50, constant_values=255)",
}


@pytest.mark.parametrize("name", list(_RENDERED))
def test_finding_a_target_imports_none_of_them(name):
    """Detection itself stays lean too, whether or not it finds the board.

    A blank frame stops at "no markers"; the target's own board goes on
    through corner interpolation, keying and assembling the detection, which
    a live preview runs on every frame the board is in view.

    The PuzzleBoard family is left out: its third-party detector imports
    scikit-image, which uses scipy for real.
    """
    skip_without_aruco2(name)
    _assert_ok(_run(
        "import numpy as np\n"
        "from pyCamSet.calibration_targets.core.target_registry import build_target\n"
        f"target = build_target({{'type': {name!r}}})\n"
        "assert not target.find_in_image(np.zeros((480, 640), np.uint8)).has_data\n"
        f"image = np.ascontiguousarray({_RENDERED[name]}, dtype=np.uint8)\n"
        "detection = target.find_in_image(image)\n"
        "assert detection.has_data and len(detection.keys) >= 9, 'board not found'\n"
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


def test_a_missing_package_is_none_only_for_an_optional_name():
    """An optional name is ``None`` without its package; any other raises."""
    getattr_ = lazy_attributes("some.module", {
        "maybe": ("not_a_package_anywhere", None),
        "surely": ("not_a_package_anywhere", None),
    }, optional={"maybe"})

    assert getattr_("maybe") is None
    with pytest.raises(ImportError):
        getattr_("surely")


def test_shape_by_faces_without_pyvista_is_as_it_was(monkeypatch):
    """Without pyvista, ``pv`` is ``None`` and ``_PYVISTA_OK`` False.

    That is what its guarded import made them, so a caller that checks
    ``getattr(shape_by_faces, "pv", None) is None`` still falls back, and
    making a shape still raises the ImportError it did.
    """
    import numpy as np
    import pyvista

    from pyCamSet.calibration_targets.core import shape_by_faces

    assert shape_by_faces.pv is pyvista
    assert shape_by_faces._PYVISTA_OK is True

    monkeypatch.setitem(sys.modules, "pyvista", None)
    assert shape_by_faces.pv is None
    assert getattr(shape_by_faces, "pv", None) is None
    assert shape_by_faces._PYVISTA_OK is False
    with pytest.raises(ImportError, match="pyvista is required"):
        shape_by_faces.make_shape("cube", 1.0)
    base = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    with pytest.raises(ImportError, match="pyvista is required"):
        shape_by_faces.make_tforms(base, "cube")
