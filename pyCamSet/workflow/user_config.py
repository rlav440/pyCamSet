"""
The little pyCamSet remembers between sessions, and where it keeps it.

Everything about a calibration is already on disk in the image folder's
workspace.  What is not is the handful of choices a person makes before any
of that exists: which folder they were working in, which target they were
calibrating.  Those are short, most-recent-first lists in one per-user
settings directory, and this is the file handling they share.

``PYCAMSET_CONFIG_DIR`` overrides the location, which is what the tests use.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: How many entries a remembered list keeps. Long enough to cover what
#: someone is moving between, short enough to stay a list rather than a
#: history.
MAX_RECENT = 10


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


def read_list(file_name: str, key: str) -> list[Any]:
    """
    The entries of one remembered list, in order.

    :param file_name: the file under :func:`config_dir` to read
    :param key: the key the entries are stored under
    :return: the entries, empty when there is no readable file
    """
    path = config_dir() / file_name
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        # A settings file is never worth failing a launch over.
        logger.debug("Could not read %s: %s", path, exc)
        return []
    entries = data.get(key) if isinstance(data, dict) else data
    return list(entries) if isinstance(entries, list) else []


def write_list(file_name: str, key: str, entries: list[Any]) -> None:
    """
    Replace one remembered list, capped at :data:`MAX_RECENT`.

    :param file_name: the file under :func:`config_dir` to write
    :param key: the key to store the entries under
    :param entries: the entries, most recent first
    """
    path = config_dir() / file_name
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({key: entries[:MAX_RECENT]}, fh, indent=2)
    except OSError as exc:
        # Read-only home, full disk, a locked profile: the GUI still works,
        # it just will not remember this next time.
        logger.debug("Could not write %s: %s", path, exc)
