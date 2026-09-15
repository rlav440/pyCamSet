"""Open3D must not be imported until something actually needs it.

``pyCamSet.utils.visualisation`` used to ``import open3d`` at module scope,
which cost every caller of the module -- including every target build, since
``cameras.camera_set`` imports this module -- roughly 1.4s whenever Open3D
happened to be installed, whether or not its renderer was ever touched.
``_open3d()`` now resolves and caches the module on first actual use.

Each check here runs in a fresh subprocess: a ``sys.modules`` assertion is
worthless once some earlier import (in this process, or an earlier test) has
already pulled Open3D in.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("open3d") is None,
    reason="open3d is not installed in this environment; nothing to prove lazy",
)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_import_does_not_pull_in_open3d():
    """A plain import of the module must leave open3d unimported."""
    result = _run(
        "import sys\n"
        "import pyCamSet.utils.visualisation\n"
        "assert 'open3d' not in sys.modules, sys.modules.get('open3d')\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_open3d_accessor_imports_and_caches():
    """Calling the accessor imports open3d and remembers it for next time."""
    result = _run(
        "import sys\n"
        "import pyCamSet.utils.visualisation as v\n"
        "mod = v._open3d()\n"
        "assert 'open3d' in sys.modules\n"
        "assert mod is not None\n"
        "assert v._OPEN3D_OK is True\n"
        "assert v._open3d() is mod\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_not_installed_message_is_unchanged():
    """Simulate a missing/broken Open3D and check the user-facing message.

    Widened from ``ImportError`` to ``Exception`` in ``_open3d()`` because a
    broken/partial Open3D install (mismatched native libs, missing CUDA, etc.)
    can fail with something other than ``ImportError``; this pins that both
    failure shapes are tolerated and produce the same guidance.
    """
    code = (
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        "def _blocking_import(name, *a, **k):\n"
        "    if name == 'open3d' or name.startswith('open3d.'):\n"
        "        raise {exc}\n"
        "    return _real_import(name, *a, **k)\n"
        "builtins.__import__ = _blocking_import\n"
        "import pyCamSet.utils.visualisation as v\n"
        "ok, msg = v.visualise_calibration_open3d({{'err': [0.0, 0.0], 'x': None}}, param_handler=None)\n"
        "assert ok is False\n"
        "expected = (\n"
        "    'Open3D is not installed. Install it with:\\n'\n"
        "    '    pip install open3d\\n'\n"
        "    'and restart the application.'\n"
        ")\n"
        "assert msg == expected, msg\n"
        "print('ok')\n"
    )
    for exc in ("ImportError('simulated missing open3d')",
                "RuntimeError('simulated broken native install')"):
        result = _run(code.format(exc=exc))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ok" in result.stdout


def test_lockbox_editor_module_import_does_not_pull_in_open3d():
    """``phase_3_lockbox_editor`` also used to import open3d at module scope.

    It is imported at GUI startup by ``phase_3_bundle_adjustment``
    (``MainWindow._build_ui``, called from ``MainWindow.__init__``), well
    before anyone opens the lockbox editor, so this needed the same lazy
    treatment as ``pyCamSet.utils.visualisation``.
    """
    result = _run(
        "import sys\n"
        "import pyCamSet.gui.phase_3_lockbox_editor as m\n"
        "assert 'open3d' not in sys.modules, sys.modules.get('open3d')\n"
        "assert m._OPEN3D_OK is None\n"
        "ok = m._ensure_open3d()\n"
        "assert ok is True\n"
        "assert 'open3d' in sys.modules\n"
        "assert m._OPEN3D_OK is True\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_bundle_adjustment_import_does_not_pull_in_open3d():
    """The GUI-startup import chain (MainWindow -> phase_3_bundle_adjustment
    -> phase_3_lockbox_editor) must not import open3d either."""
    result = _run(
        "import sys\n"
        "import pyCamSet.gui.phase_3_bundle_adjustment\n"
        "assert 'open3d' not in sys.modules, sys.modules.get('open3d')\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout
