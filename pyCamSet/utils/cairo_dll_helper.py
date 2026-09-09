'''
Purpose: Provide a portable, one-time Cairo native-library loader so that
         `cairosvg` / `cairocffi` can find `cairo.dll` (or the platform
         equivalent) without every caller hardcoding a conda-env path.
Status: Active. Added to fix the headless-Cairo DLL quirk centrally.
Future: If pyCamSet ever drops the cairosvg dependency, this module can go.
'''

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Guard against running the registration more than once per process.
_cairo_dll_registered: bool = False


def _conda_library_bin() -> Optional[Path]:
    '''
    Return the `Library/bin` directory of the running Python environment on
    Windows conda/venv installs, or `None` when that path does not exist.

    On a conda environment the layout is `<env>/Library/bin` (the native Cairo
    DLL ships there).  On a standard CPython venv there is no `Library` tree so
    we return `None` and let the caller fall back to the system PATH.
    '''
    candidate = Path(sys.prefix) / "Library" / "bin"
    return candidate if candidate.is_dir() else None


def ensure_cairo_dll_available() -> bool:
    '''
    Make the native Cairo library discoverable for `cairocffi`/`cairosvg`.

    Tries a bare `import cairosvg` first; if that already works (e.g. the DLL is
    already on the search path, or we are not on Windows) this is a silent
    no-op.  Only on failure do we derive the conda `Library/bin` path relative
    to the running interpreter, register it via `os.add_dll_directory` (Python
    3.8+ DLL isolation) and prepend it to `PATH` (fallback for libraries that
    still honour the classic search path), then retry the import.

    Returns True when cairosvg is importable, False otherwise.
    '''
    global _cairo_dll_registered

    if _cairo_dll_registered:
        return True

    # Fast path: cairo already loads fine — nothing to do.
    try:
        import cairosvg  # noqa: F401
        _cairo_dll_registered = True
        return True
    except Exception:
        pass  # Fall through to the DLL-directory registration below.

    # Only the Windows conda layout needs this workaround.
    lib_bin = _conda_library_bin()
    if lib_bin is not None:
        str_path = str(lib_bin)
        # Prepend to PATH so legacy DLL searches (used by some CFFI backends) find it.
        os.environ["PATH"] = str_path + os.pathsep + os.environ.get("PATH", "")
        # Add the directory via the modern 3.8+ API so isolated DLL loading finds it.
        try:
            os.add_dll_directory(str_path)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            # add_dll_directory may be absent on non-Windows or very old builds.
            pass

    # Retry the import so we can report success/failure honestly.
    try:
        import cairosvg  # noqa: F401
        _cairo_dll_registered = True
        return True
    except Exception:
        return False


# Run once at import time so any module that imports this helper (or is
# imported after it) gets a working cairosvg without an explicit call.
ensure_cairo_dll_available()
