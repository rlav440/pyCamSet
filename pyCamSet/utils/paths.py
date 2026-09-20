"""
Paths in the form the filesystem will accept.
"""
from __future__ import annotations

import os
from pathlib import Path


def long_path(path: Path | str) -> Path:
    """*path*, absolute and safe to open however long it is.

    Windows caps a path at 260 characters unless it is in extended-length
    (``\\\\?\\``) form, which requires an absolute, normalised path and then
    suppresses further normalisation -- so append only plain segments to the
    result. Outside Windows this is ``abspath``.
    """
    absolute = os.path.abspath(os.fspath(path))
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)
