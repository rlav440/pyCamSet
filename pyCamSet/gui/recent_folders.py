"""
The image folders this machine has calibrated before.

Everything the GUI remembers about a calibration lives in
``<image folder>/.pycamset_workspace``: the runs, their artifacts, and the
handoff between phases.  All of it is found again by pointing the GUI at
the same image folder -- and the image folder itself was the one thing
nothing recorded, so a reopened GUI started blank with no way to discover
where it had been.

This is that one thing.  It is deliberately not a session file: no window
geometry, no half-filled forms, no open tab.  A folder is a place the work
lives, and the rest is rebuilt from what the runs already wrote down.

Stored as JSON under the platform's per-user configuration directory.
``PYCAMSET_CONFIG_DIR`` overrides that, which is what the tests use.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: How many folders to keep. Long enough to cover the projects someone is
#: moving between, short enough to stay a list rather than a history.
MAX_RECENT = 10

_FILE_NAME = "recent_folders.json"


def config_dir() -> Path:
    """
    Where this machine keeps pyCamSet's per-user settings.

    :return: the directory, which may not exist yet
    """
    override = os.environ.get("PYCAMSET_CONFIG_DIR")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "pyCamSet"
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
        return root / "pyCamSet"
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "pyCamSet"


def recent_folders_file() -> Path:
    """:return: the file the folder list is stored in"""
    return config_dir() / _FILE_NAME


def _read() -> list[str]:
    path = recent_folders_file()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        # A settings file is never worth failing a launch over.
        logger.debug("Could not read %s: %s", path, exc)
        return []
    folders = data.get("folders") if isinstance(data, dict) else data
    if not isinstance(folders, list):
        return []
    return [str(f) for f in folders if isinstance(f, (str, Path))]


def _write(folders: list[str]) -> None:
    path = recent_folders_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"folders": folders}, fh, indent=2)
    except OSError as exc:
        # Read-only home, full disk, a locked profile: the GUI still works,
        # it just will not remember this next time.
        logger.debug("Could not write %s: %s", path, exc)


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
    _write([str(resolved), *kept][:MAX_RECENT])


def forget_folder(folder: Path | str) -> None:
    """
    Drop a folder from the list.

    :param folder: the folder to forget
    """
    try:
        resolved = Path(folder).expanduser().resolve()
    except OSError:
        return
    _write([f for f in _read() if _differs(f, resolved)])


def _differs(candidate: str, resolved: Path) -> bool:
    """Whether ``candidate`` names somewhere other than ``resolved``."""
    try:
        return Path(candidate).expanduser().resolve() != resolved
    except OSError:
        return True
