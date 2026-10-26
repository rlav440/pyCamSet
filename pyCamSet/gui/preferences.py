"""Purpose: Persist presentation-only GUI choices in a versioned user config file.
Status: Active; stores preferences outside workspaces and preserves malformed files.
Future: Add new presentation keys only with schema validation and restart tests.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

_SCHEMA = "pycamset.gui-preferences"
_VERSION = 1
#: A visual id: two or more lower-case ``:``-separated segments, such as
#: ``phase1:detection-montage`` or ``diagnostic:phase2:d2.6-per-image-reprojection``.
_PRESET_KEY = re.compile(r"[a-z0-9][a-z0-9._-]*(?::[a-z0-9][a-z0-9._-]*)+\Z")
_DEFAULTS: dict[str, Any] = {
    "info_enabled": True,
    "terminal_visible": True,
    "export_presets": {},
}


def initialise_application_identity(application) -> None:
    """Set the Qt identity that names the per-user config directory."""
    application.setOrganizationName("pyCamSet")
    application.setApplicationName("pyCamSet")


def config_directory() -> Path:
    """Return the per-user application config directory.

    Resolution order: ``PYCAMSET_CONFIG_DIR``, then Qt's ``AppConfigLocation``,
    then :func:`pyCamSet.workflow.user_config.config_dir`.

    Qt is consulted whenever it is importable, whether or not a ``QApplication``
    exists yet.  Qt builds ``AppConfigLocation`` from the same
    organisation/application statics a launched GUI sets, so a GUI-less process
    -- the education sidecar, a helper script -- resolves the directory the GUI
    itself uses rather than a different one.  Those statics are filled in below
    when they are still empty: an empty pair makes Qt answer a bare
    ``<Local AppData>`` (or the interpreter's own name) instead of pyCamSet's
    own folder, which is not what any caller wants.

    The fallback is the per-user setting the recent-folders and recent-targets
    memory uses.  The two are *different* directories, and which differ is
    platform-dependent.  Observed on Windows (PySide6 6.11): Qt answers
    ``%LOCALAPPDATA%\\pyCamSet\\pyCamSet`` while :func:`config_dir` answers
    ``%APPDATA%\\pyCamSet`` (Roaming).  On Linux Qt's ``AppConfigLocation`` is
    under ``$XDG_CONFIG_HOME`` (default ``~/.config``) like :func:`config_dir`,
    but Qt appends organisation and application names, so with the identity set
    here it is expected to be one level deeper (``.../pyCamSet/pyCamSet``).  On
    macOS Qt's ``AppConfigLocation`` is documented as under
    ``~/Library/Preferences``, whereas :func:`config_dir` uses
    ``~/Library/Application Support``.  The Linux and macOS statements come from
    Qt's documentation and were not run on those systems.  Nowhere should the two
    be assumed to agree, so a preferences file written through one is not visible
    to the other.  The fallback is therefore reached only where Qt is genuinely
    unavailable: PySide6 not installed (the lean install of requirements_core.txt,
    which must still be able to import the education sidecar) or installed but
    unable to load its Qt libraries.  Every branch honours the override.
    """
    override = os.environ.get("PYCAMSET_CONFIG_DIR")
    if override:
        return Path(override)
    try:
        from PySide6.QtCore import QCoreApplication, QStandardPaths
    except ImportError:
        # No GUI toolkit: the education sidecar must still import.  This is the
        # Roaming fallback noted above, which is why the two must not be
        # described as agreeing.  Imported here because ``pyCamSet.workflow``
        # pulls in every phase, which the GUI-less preference read need not pay for.
        from pyCamSet.workflow.user_config import config_dir as _fallback_config_dir
        return _fallback_config_dir()
    # Qt resolves the application directory from these statics, and names a
    # directory only after they are set.  Setting them here is the same call
    # :func:`initialise_application_identity` makes on a real launch, so a
    # process that never builds a window still lands in the GUI's own
    # directory.  An identity deliberately set by someone else is left alone.
    if not QCoreApplication.organizationName():
        QCoreApplication.setOrganizationName("pyCamSet")
    if not QCoreApplication.applicationName():
        QCoreApplication.setApplicationName("pyCamSet")
    location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation)
    if not location:
        raise OSError("Qt did not provide an application configuration directory")
    return Path(location)


def preferences_path() -> Path:
    """Return the explicit JSON preferences file path."""
    return config_directory() / "preferences.json"


def _validated(document: object) -> dict[str, Any]:
    if not isinstance(document, dict) or set(document) != {"schema", "version", "preferences"}:
        raise ValueError("Preferences file has missing or unknown fields")
    if document["schema"] != _SCHEMA or type(document["version"]) is not int or document["version"] != _VERSION:
        raise ValueError("Unsupported preferences schema or version")
    values = document["preferences"]
    if not isinstance(values, dict) or set(values) != set(_DEFAULTS):
        raise ValueError("Preferences contain missing or unknown settings")
    if type(values["info_enabled"]) is not bool or type(values["terminal_visible"]) is not bool:
        raise ValueError("Boolean preferences must be true or false")
    presets = values["export_presets"]
    if not isinstance(presets, dict) or any(
            not isinstance(key, str) or not _PRESET_KEY.fullmatch(key)
            or type(index) is not int or index not in range(3)
            for key, index in presets.items()):
        raise ValueError("Export preset selections are invalid")
    return {"info_enabled": values["info_enabled"],
            "terminal_visible": values["terminal_visible"],
            "export_presets": dict(presets)}


class Preferences:
    """Validated presentation preferences with recoverable atomic writes."""

    def __init__(self) -> None:
        self.path = preferences_path()
        self.values = dict(_DEFAULTS)
        self.values["export_presets"] = {}
        self.load_error: str | None = None
        self._preserve_corrupt = False
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.values = _validated(raw)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self.load_error = str(exc)
                self._preserve_corrupt = True

    def set(self, key: str, value: Any) -> None:
        previous = json.loads(json.dumps(self.values))
        if key in ("info_enabled", "terminal_visible"):
            if type(value) is not bool:
                raise ValueError(f"{key} must be boolean")
            self.values[key] = value
        elif key.startswith("export_preset:"):
            if type(value) is not int or value not in range(3):
                raise ValueError("Export preset index must be 0, 1 or 2")
            if not _PRESET_KEY.fullmatch(key.removeprefix("export_preset:")):
                raise ValueError(f"Invalid visual id for an export preset: {key}")
            self.values["export_presets"][key.removeprefix("export_preset:")] = value
        else:
            raise ValueError(f"Unknown presentation preference: {key}")
        try:
            self.save()
        except (OSError, ValueError):
            self.values = previous
            raise

    def save(self) -> None:
        document = {"schema": _SCHEMA, "version": _VERSION, "preferences": self.values}
        _validated(document)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._preserve_corrupt and self.path.exists():
            backup = self.path.with_name("preferences.json.corrupt")
            if not backup.exists():
                shutil.copy2(self.path, backup)
            self._preserve_corrupt = False
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n", dir=self.path.parent,
                    prefix=".preferences-", suffix=".tmp", delete=False) as stream:
                temp_name = stream.name
                json.dump(document, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)


def get_preferences() -> Preferences:
    """Share one in-memory preference document across all GUI widgets."""
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance()
    if application is None:
        raise RuntimeError("Preferences require an active QApplication")
    instance = getattr(application, "_pycamset_preferences", None)
    if instance is None:
        instance = Preferences()
        application._pycamset_preferences = instance
    return instance


def preference_index(preferences: Preferences, key: str) -> int:
    return preferences.values["export_presets"].get(key, 0)


def bind_export_preset(combo, key: str) -> None:
    """Restore a fixed three-option presentation preset and persist changes."""
    if not _PRESET_KEY.fullmatch(key):
        raise ValueError(f"Invalid visual id for an export preset: {key}")
    preferences = get_preferences()
    combo.setCurrentIndex(preference_index(preferences, key))
    def persist(index: int) -> None:
        try:
            preferences.set(f"export_preset:{key}", index)
        except (OSError, ValueError) as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(combo, "Preference not saved", str(exc))
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(preference_index(preferences, key))
            combo.blockSignals(blocked)

    combo.currentIndexChanged.connect(persist)
