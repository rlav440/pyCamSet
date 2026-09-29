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

from PySide6.QtCore import QStandardPaths

_SCHEMA = "pycamset.gui-preferences"
_VERSION = 1
_PRESET_KEY = re.compile(
    r"(?:figure:[a-z0-9]+(?:-[a-z0-9]+)*|phase[1-4]:(?:detection-montage|3d-export|assessment-export))\Z"
)
_DEFAULTS: dict[str, Any] = {
    "info_enabled": True,
    "terminal_visible": True,
    "export_presets": {},
}


def initialise_application_identity(application) -> None:
    """Set stable Qt identity and migrate old generic per-visual style files."""
    old_location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation)
    application.setOrganizationName("pyCamSet")
    application.setApplicationName("pyCamSet")
    if os.environ.get("PYCAMSET_CONFIG_DIR"):
        # The established override is used by tests and explicit portable setups.
        return
    new_location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation)
    if not old_location or not new_location or Path(old_location) == Path(new_location):
        return
    old_styles = Path(old_location) / "visual-styles"
    new_styles = Path(new_location) / "visual-styles"
    if old_styles.is_dir():
        for source in old_styles.glob("*.json"):
            destination = new_styles / source.name
            if not destination.exists():
                try:
                    new_styles.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                except OSError:
                    # Keep the legacy file intact; callers continue with defaults.
                    continue


def config_directory() -> Path:
    """Return Qt's stable, per-user application config directory."""
    override = os.environ.get("PYCAMSET_CONFIG_DIR")
    if override:
        return Path(override)
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
    preferences = get_preferences()
    combo.setCurrentIndex(preference_index(preferences, key))
    def persist(index: int) -> None:
        try:
            preferences.set(f"export_preset:{key}", index)
        except OSError as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(combo, "Preference not saved", str(exc))
            combo.setCurrentIndex(preference_index(preferences, key))

    combo.currentIndexChanged.connect(persist)
