'''
:Purpose: Provide a portable, one-time Cairo native-library loader so that
    ``cairosvg`` / ``cairocffi`` can find ``cairo.dll`` (or the platform
    equivalent) without every caller hardcoding a conda-env path.
:Status: Active. Added to fix the headless-Cairo DLL quirk centrally.
:Future: If pyCamSet ever drops the cairosvg dependency, this module can go.
'''

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Guard against running the registration more than once per process.
_cairo_dll_registered: bool = False

#: The file names ``cairocffi`` actually loads on Windows: ``cairo-2``, ``cairo``
#: and ``libcairo-2`` are searched for on ``PATH``, then ``libcairo-2.dll``
#: directly.  A directory is only worth registering if it holds one of these;
#: ``cairo-gobject.dll`` and ``cairo-script-interpreter.dll`` sit beside Cairo
#: in a conda ``Library/bin`` but are not Cairo.
_CAIRO_DLL_NAMES: tuple[str, ...] = ("cairo-2.dll", "cairo.dll", "libcairo-2.dll")


def _register_dll_dir(lib_bin: Path) -> None:
    '''
    Make the running interpreter's own `lib_bin` searchable for native libraries.

    The directory is prepended to `PATH` (the fallback for libraries that still
    honour the classic search path) and added through the Python 3.8+
    `os.add_dll_directory` API (the modern, isolated DLL search).  This is the
    interpreter's own conda layout, which is trusted; the wider search over other
    installations uses :func:`_try_cairosvg_from` instead, which keeps nothing
    it did not need.
    '''
    str_path = str(lib_bin)
    # Prepend to PATH so legacy DLL searches (used by some CFFI backends) find it.
    os.environ["PATH"] = str_path + os.pathsep + os.environ.get("PATH", "")
    # Add the directory via the modern 3.8+ API so isolated DLL loading finds it.
    try:
        os.add_dll_directory(str_path)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        # add_dll_directory may be absent on non-Windows or very old builds.
        pass


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


def _conda_roots() -> list[Path]:
    '''
    Conda environments and installation roots this process can see, best first.

    The interpreter a venv was built on is not recorded in the venv, and a
    headless process need not have been started from an activated environment,
    so the roots are read from the places Windows actually leaves them: the
    activation variables, the `condabin` (or an environment's `Scripts`) on
    `PATH`, this interpreter's base prefix, and conda's own registry of
    environments.

    Only the directories are returned; whether any of them carries Cairo is decided
    later, by looking at the directory itself.
    '''
    roots: list[Path] = []

    def add_one(path: Path) -> None:
        # A drive root ("C:\") is never a conda installation, and searching
        # one would register whatever happens to sit in `C:\Library\bin`.
        if path.parent == path:
            return
        if path not in roots:
            roots.append(path)

    def add(value: object) -> None:
        root = Path(str(value))
        # An environment may carry Cairo itself, so it is searched as well as
        # the installation it belongs to (`<root>/envs/<name>` -> `<root>`).
        add_one(root)
        if root.parent.name.lower() == "envs":
            add_one(root.parent.parent)

    for var in ("CONDA_ROOT", "CONDA_PREFIX"):
        value = os.environ.get(var)
        if value:
            add(value)

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        part = Path(entry)
        # PATH may name `conda.bat`/`conda.exe` itself rather than its directory.
        if part.name.lower().startswith("conda."):
            part = part.parent
        if part.name.lower() == "condabin":
            add(part.parent)
        elif part.parent.name.lower() == "envs":
            add(part)
        elif part.name.lower() in ("scripts", "bin") and (part.parent / "condabin").is_dir():
            add(part.parent)

    base = Path(sys.base_prefix)
    if (base / "condabin").is_dir() or (base / "Library" / "bin").is_dir():
        add(base)

    # conda keeps a plain list of every environment it knows; it is the only
    # place a second installation outside PATH is recorded.  Reading it is
    # best effort: a missing or unreadable file just means fewer candidates.
    try:
        for line in (Path.home() / ".conda" / "environments.txt").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                add(line.strip())
    except (OSError, RuntimeError):
        # RuntimeError: pathlib cannot resolve "~" when the home variables are
        # absent, as they are in some service-like Windows environments.
        pass

    return roots


def _library_bin_dirs() -> list[Path]:
    '''
    Every `Library/bin` this machine is likely to keep the native Cairo in,
    best first: the running interpreter's own conda layout, then the layouts of
    the conda installations discoverable from the environment.
    '''
    dirs: list[Path] = []

    def add(path: Path) -> None:
        if path.is_dir() and path not in dirs:
            dirs.append(path)

    add(Path(sys.prefix) / "Library" / "bin")
    if sys.platform == "win32":
        # Only the Windows conda layout needs this workaround.
        for root in _conda_roots():
            add(root / "Library" / "bin")
    return dirs


def ensure_cairo_dll_available() -> bool:
    '''
    Make the native Cairo library discoverable for `cairocffi`/`cairosvg`.

    Tries a bare `import cairosvg` first; if that already works (e.g. the DLL is
    already on the search path, or we are not on Windows) this is a silent
    no-op.  Only on failure do we derive the conda `Library/bin` path relative
    to the running interpreter, register it, and retry the import.  It does not
    search other conda installations; that wider search is
    :func:`_cairosvg_from_conda_roots`, run by :func:`cairosvg_or_explain`.

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

    lib_bin = _conda_library_bin()
    if lib_bin is not None:
        _register_dll_dir(lib_bin)

    # Retry the import so we can report success/failure honestly.
    try:
        import cairosvg  # noqa: F401
        _cairo_dll_registered = True
        return True
    except Exception:
        return False


def _has_cairo_dll(lib_bin: Path) -> bool:
    '''Whether `lib_bin` holds a DLL under a name `cairocffi` loads.'''
    return any((lib_bin / name).is_file() for name in _CAIRO_DLL_NAMES)


def _try_cairosvg_from(lib_bin: Path) -> Optional[object]:
    '''
    Import `cairosvg` with `lib_bin` searchable, or leave the process as found.

    `cairocffi` looks for Cairo through `PATH`, so `PATH` is extended for the
    duration of the import only and restored afterwards, whether or not the
    import worked: Cairo's own dependencies are loaded by then, and a whole
    `Library/bin` left on `PATH` would shadow DLLs that later code loads.  The
    handle `os.add_dll_directory` returns is kept only on success (the loaded
    library may resolve a dependency lazily) and closed on failure.
    '''
    handle = os.add_dll_directory(str(lib_bin))  # type: ignore[attr-defined]
    saved_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(lib_bin) + os.pathsep + saved_path
    try:
        import cairosvg
    except Exception:
        handle.close()
        return None
    finally:
        os.environ["PATH"] = saved_path
    return cairosvg


def _cairosvg_from_conda_roots() -> Optional[object]:
    '''
    Last-resort import of `cairosvg`: try the `Library/bin` of every conda
    environment and installation this process can see, returning the module on
    the first directory that works, else `None`.

    This is not run by :func:`ensure_cairo_dll_available` or at import time.  It
    runs from :func:`cairosvg_or_explain`, and only after a bare
    `import cairosvg` has raised `OSError` on Windows, i.e. when a target is
    actually being drawn and Cairo could not be found.

    A venv built on a conda interpreter (uv, virtualenv) has no `Library` tree
    of its own, so `ensure_cairo_dll_available` has nothing of the interpreter's
    to register; and a process started outside an activated environment (a
    scheduled task, a service, a test runner) does not inherit the base
    installation's `Library/bin` on `PATH` either.  Between them that leaves
    `cairosvg` importable but unable to find Cairo.

    Only a directory holding a DLL named as `cairocffi` loads it
    (:data:`_CAIRO_DLL_NAMES`) is tried, one at a time, and a directory that does
    not make the import work is unregistered again.
    '''
    for lib_bin in _library_bin_dirs():
        if not _has_cairo_dll(lib_bin):
            continue
        cairosvg = _try_cairosvg_from(lib_bin)
        if cairosvg is not None:
            return cairosvg
    return None


# Run once at import time so any module that imports this helper (or is
# imported after it) gets a working cairosvg without an explicit call.
ensure_cairo_dll_available()


def cairosvg_or_explain():
    """The ``cairosvg`` module, or an OSError saying how to install cairo.

    Imported here rather than at module scope so that only a target actually
    being drawn pays for it.  When the import fails because the native library
    cannot be found, the wider conda search above gets one chance to locate it
    before the error is reported: a machine that has a conda installation with
    Cairo in it should draw, whether or not that installation is the running
    interpreter's or is on this process's `PATH`.
    """
    try:
        import cairosvg
    except OSError as err:
        cairosvg = _cairosvg_from_conda_roots() if sys.platform == "win32" else None
        if cairosvg is None:
            raise OSError(
                f"{err}\n\n"
                "pyCamSet's target drawing requires the native 'cairo' library, "
                "which cairosvg needs but pip cannot install on its own.\n"
                "Install it for your platform, then re-import pyCamSet:\n"
                "  - conda (Windows/Linux/macOS):  conda install -c conda-forge cairo\n"
                "  - Debian/Ubuntu:                 apt install libcairo2\n"
                "  - macOS (Homebrew):              brew install cairo\n"
                "  - Windows (no conda):            install GTK/cairo and put the DLL on PATH"
            ) from err
    return cairosvg
