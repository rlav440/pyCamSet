"""
The image folders this machine has calibrated before.

Everything pyCamSet remembers about a calibration lives in
``<image folder>/.pycamset_workspace``: the runs, their artifacts, and the
handoff between phases.  All of it is found again by pointing the GUI at
the same image folder -- and the image folder itself was the one thing
nothing recorded, so a reopened GUI started blank with no way to discover
where it had been.

This is that one thing.  It is deliberately not a session file: no window
geometry, no half-filled forms, no open tab.  A folder is a place the work
lives, and the rest is rebuilt from what the runs already wrote down.

Stored through :mod:`pyCamSet.workflow.user_config`, alongside the other
thing a person chooses before any of the work exists: see
:mod:`pyCamSet.workflow.recent_targets`.
"""
from __future__ import annotations

from pathlib import Path

from pyCamSet.workflow.user_config import (
    MAX_RECENT, config_dir, read_list, write_list)

_FILE_NAME = "recent_folders.json"
_KEY = "folders"


def recent_folders_file() -> Path:
    """:return: the file the folder list is stored in"""
    return config_dir() / _FILE_NAME


def _read() -> list[str]:
    return [str(f) for f in read_list(_FILE_NAME, _KEY)
            if isinstance(f, (str, Path))]


def load_recent_folders(existing_only: bool = True) -> list[Path]:
    """
    The folders last worked in, most recent first.

    :param existing_only: drop entries that are no longer directories --
        removable drives and deleted projects otherwise accumulate
    :return: the folders
    """
    folders = [Path(f) for f in _read()]
    if existing_only:
        folders = [f for f in folders if f.is_dir()]
    return folders[:MAX_RECENT]


def remember_folder(folder: Path | str) -> None:
    """
    Put a folder at the top of the list.

    Called where the folder is known to be real -- a validated selection,
    or a run that has just been written into its workspace -- rather than
    on every keystroke that happens to name a directory.

    :param folder: the image folder to remember
    """
    if not folder:
        return
    try:
        resolved = Path(folder).expanduser().resolve()
    except OSError:
        return

    kept = [f for f in _read() if _differs(f, resolved)]
    write_list(_FILE_NAME, _KEY, [str(resolved), *kept])


def forget_folder(folder: Path | str) -> None:
    """
    Drop a folder from the list.

    :param folder: the folder to forget
    """
    try:
        resolved = Path(folder).expanduser().resolve()
    except OSError:
        return
    write_list(_FILE_NAME, _KEY, [f for f in _read() if _differs(f, resolved)])


def _differs(candidate: str, resolved: Path) -> bool:
    """Whether ``candidate`` names somewhere other than ``resolved``."""
    try:
        return Path(candidate).expanduser().resolve() != resolved
    except OSError:
        return True
