"""Store tolerant education preferences, including the mode gate.

``education_enabled`` is the education mode flag: a persisted
per-user preference rather than a build distinction (pyCamSet has only one
build), off by default until the native acceptance card passes.  It lives here
because F1a already owns this sidecar and a separate file would be a second
preference contract for one flag.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pyCamSet.gui.preferences import config_directory

_SCHEMA = "pycamset.education-preferences"
_VERSION = 1
_DEFAULTS: dict[str, bool] = {"shortcuts_enabled": True, "reduce_motion": False}

#: The education mode flag.
#:
#: Deliberately *not* added to :data:`_DEFAULTS`: that mapping is the set of
#: presentation keys :func:`load_preferences` returns, and widening it would
#: change the value every caller (and F1a's own tests) receives for an existing
#: or absent file.  The mode has its own reader instead, which is strictly
#: fail-closed, and it is added to what :func:`save_preferences` may write so a
#: settings surface can toggle it later.
_MODE_KEY = "education_enabled"
_WRITABLE: frozenset[str] = frozenset(_DEFAULTS) | {_MODE_KEY}


def preferences_path() -> Path:
    """Return the education-only sidecar beside ``preferences.json``."""
    return config_directory() / "education_preferences.json"


def education_mode_enabled(path: Path | None = None) -> bool:
    """Whether the education mode is switched on for this user.

    Reads the same tolerant sidecar as :func:`load_preferences`, but only the
    mode key.  Fail-closed: an absent file, a malformed document, a missing key
    or a value that is not literally ``true`` all read as *off*, which is the
    documented default.
    """
    target = path or preferences_path()
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(document, dict):
        return False
    stored = document.get("preferences")
    if not isinstance(stored, dict):
        return False
    return stored.get(_MODE_KEY) is True


def load_preferences(path: Path | None = None) -> dict[str, Any]:
    """Load recognised values; ignore unknown/missing fields and bad types."""
    target = path or preferences_path()
    values: dict[str, Any] = dict(_DEFAULTS)
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return values
    if not isinstance(document, dict):
        return values
    stored = document.get("preferences", {})
    if not isinstance(stored, dict):
        return values
    for key, default in _DEFAULTS.items():
        if type(stored.get(key)) is type(default):
            values[key] = stored[key]
    return values


def save_preferences(updates: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Merge known updates into existing JSON, retaining keys this version does not know."""
    target = path or preferences_path()
    if not isinstance(updates, dict) or set(updates) - _WRITABLE:
        raise ValueError("Only declared education preference keys may be updated")
    for key, value in updates.items():
        if type(value) is not bool:
            raise ValueError(f"{key} must be boolean")

    try:
        document = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            document = {}
    except (OSError, ValueError, TypeError):
        document = {}
    stored = document.get("preferences")
    if not isinstance(stored, dict):
        stored = {}
    stored.update(updates)
    document.update({"schema": _SCHEMA, "version": _VERSION, "preferences": stored})

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=target.parent,
                prefix=".education-preferences-", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return load_preferences(target)
