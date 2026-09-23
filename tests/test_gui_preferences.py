"""Purpose: Verify versioned GUI preference storage and recovery semantics.
Status: Active; tests use isolated paths and presentation-only values.
Future: Extend fresh-process coverage as preference keys are added.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyCamSet.gui import preferences


def test_preferences_round_trip_and_defaults(tmp_path, monkeypatch):
    """Defaults and supported presentation selections survive reconstruction."""
    path = tmp_path / "preferences.json"
    monkeypatch.setattr(preferences, "preferences_path", lambda: path)
    first = preferences.Preferences()
    assert first.values == {"info_enabled": True, "terminal_visible": True, "export_presets": {}}
    first.set("info_enabled", False)
    first.set("terminal_visible", False)
    first.set("export_preset:phase3:3d-export", 2)
    restored = preferences.Preferences()
    assert restored.values == {
        "info_enabled": False,
        "terminal_visible": False,
        "export_presets": {"phase3:3d-export": 2},
    }
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema"] == "pycamset.gui-preferences"
    assert document["version"] == 1


def test_invalid_or_corrupt_settings_use_defaults_without_silent_loss(tmp_path, monkeypatch):
    """Malformed settings are rejected and preserved before an edit replaces them."""
    path = tmp_path / "preferences.json"
    original = "{not-json\n"
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(preferences, "preferences_path", lambda: path)
    loaded = preferences.Preferences()
    assert loaded.values["info_enabled"] is True
    assert loaded.load_error
    assert path.read_text(encoding="utf-8") == original
    loaded.set("info_enabled", False)
    assert path.with_name("preferences.json.corrupt").read_text(encoding="utf-8") == original
    assert preferences.Preferences().values["info_enabled"] is False


def test_schema_validation_rejects_unknown_or_invalid_preferences(tmp_path, monkeypatch):
    """Unsupported schema content is not partially accepted."""
    path = tmp_path / "preferences.json"
    monkeypatch.setattr(preferences, "preferences_path", lambda: path)
    invalid = {
        "schema": "pycamset.gui-preferences", "version": 1,
        "preferences": {"info_enabled": 1, "terminal_visible": True, "export_presets": {}},
    }
    path.write_text(json.dumps(invalid), encoding="utf-8")
    loaded = preferences.Preferences()
    assert loaded.load_error
    assert loaded.values["info_enabled"] is True
    with pytest.raises(ValueError):
        loaded.set("export_preset:bad", 3)
    path.write_text(json.dumps({
        "schema": "pycamset.gui-preferences", "version": 1,
        "preferences": {"info_enabled": True, "terminal_visible": True,
                        "export_presets": {"C:/private/input": 1}},
    }), encoding="utf-8")
    assert preferences.Preferences().load_error


def test_failed_atomic_write_keeps_old_file_and_memory(tmp_path, monkeypatch):
    """A failed replacement does not damage either the last file or live values."""
    path = tmp_path / "preferences.json"
    monkeypatch.setattr(preferences, "preferences_path", lambda: path)
    loaded = preferences.Preferences()
    loaded.set("info_enabled", False)
    original = path.read_bytes()
    monkeypatch.setattr(preferences.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("read-only")))
    with pytest.raises(OSError, match="read-only"):
        loaded.set("terminal_visible", False)
    assert path.read_bytes() == original
    assert loaded.values["terminal_visible"] is True


def test_application_identity_migrates_legacy_styles_without_deleting_source(tmp_path, monkeypatch):
    """Existing generic-config visual styles are copied, not moved or overwritten."""
    from PySide6.QtCore import QStandardPaths
    from PySide6.QtWidgets import QApplication

    monkeypatch.delenv("PYCAMSET_CONFIG_DIR", raising=False)
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("")
    app.setApplicationName("python")
    old_root = tmp_path / "legacy"
    new_root = tmp_path / "pyCamSet"
    legacy_style = old_root / "visual-styles" / "style.json"
    legacy_style.parent.mkdir(parents=True)
    legacy_style.write_text("legacy-style", encoding="utf-8")

    def resolved_location(_location):
        return str(new_root if app.organizationName() == "pyCamSet" else old_root)

    monkeypatch.setattr(QStandardPaths, "writableLocation", resolved_location)
    preferences.initialise_application_identity(app)
    migrated = new_root / "visual-styles" / "style.json"
    assert migrated.read_text(encoding="utf-8") == "legacy-style"
    assert legacy_style.read_text(encoding="utf-8") == "legacy-style"


def test_fresh_gui_process_persists_preferences_outside_repository(tmp_path):
    """A second real GUI process reads the first process's external config."""
    repo = Path(__file__).resolve().parents[1]
    isolated_local = tmp_path / "Local"
    isolated_roaming = tmp_path / "Roaming"
    isolated_config = tmp_path / "QtConfig" / "pyCamSet"
    env = os.environ.copy()
    env.update({
        "QT_QPA_PLATFORM": "offscreen",
        "PYVISTA_OFF_SCREEN": "true",
        "LOCALAPPDATA": str(isolated_local),
        "APPDATA": str(isolated_roaming),
        "PYCAMSET_CONFIG_DIR": str(isolated_config),
        "PYTHONPATH": str(repo),
    })
    writer = """
import json, sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings, QStandardPaths
_writable = QStandardPaths.writableLocation
QStandardPaths.writableLocation = lambda location: sys.argv[2] if location == QStandardPaths.StandardLocation.AppConfigLocation else _writable(location)
import pyCamSet.gui.main_window as main_window_module
main_window_module.QSettings = lambda *_args: QSettings(sys.argv[1], QSettings.Format.IniFormat)
from pyCamSet.gui.main_window import PyCamSetApp
app = QApplication([])
window = PyCamSetApp()
window._info_cb.setChecked(False)
window._terminal_cb.setChecked(False)
window._theme_combo.setCurrentText('Sepia')
from pyCamSet.gui.preferences import config_directory, preferences_path
print(json.dumps({'config': str(config_directory()), 'prefs': str(preferences_path()), 'theme': window._theme_combo.currentText()}))
window.close()
"""
    reader = """
import json
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings, QStandardPaths
_writable = QStandardPaths.writableLocation
QStandardPaths.writableLocation = lambda location: sys.argv[2] if location == QStandardPaths.StandardLocation.AppConfigLocation else _writable(location)
import pyCamSet.gui.main_window as main_window_module
main_window_module.QSettings = lambda *_args: QSettings(sys.argv[1], QSettings.Format.IniFormat)
from pyCamSet.gui.main_window import PyCamSetApp
from pyCamSet.gui.preferences import config_directory, preferences_path
app = QApplication([])
window = PyCamSetApp()
print(json.dumps({'config': str(config_directory()), 'prefs': str(preferences_path()), 'info': window._info_cb.isChecked(), 'terminal': window._terminal_cb.isChecked(), 'theme': window._theme_combo.currentText()}))
window.close()
"""
    theme_file = str(tmp_path / "theme.ini")
    first = subprocess.run([sys.executable, "-c", writer, theme_file, str(isolated_config)], cwd=repo, env=env,
                           check=True, capture_output=True, text=True, encoding="utf-8")
    second = subprocess.run([sys.executable, "-c", reader, theme_file, str(isolated_config)], cwd=repo, env=env,
                            check=True, capture_output=True, text=True, encoding="utf-8")
    first_data = json.loads(first.stdout.splitlines()[-1])
    second_data = json.loads(second.stdout.splitlines()[-1])
    assert Path(first_data["config"]) == isolated_config
    assert not Path(first_data["config"]).is_relative_to(repo)
    assert first_data["config"] == second_data["config"]
    assert second_data["info"] is False
    assert second_data["terminal"] is False
    assert second_data["theme"] == "Sepia"


def test_qt_identity_resolves_an_app_specific_external_directory(tmp_path):
    """The unoverridden Qt application-config path is namespaced and external."""
    repo = Path(__file__).resolve().parents[1]
    probe = """
import json
from PySide6.QtWidgets import QApplication
from pyCamSet.gui.preferences import config_directory
app = QApplication([])
app.setOrganizationName('pyCamSet')
app.setApplicationName('pyCamSet')
print(json.dumps({'config': str(config_directory()), 'application': app.applicationName(), 'organisation': app.organizationName()}))
"""
    env = os.environ.copy()
    env.update({"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(repo)})
    env.pop("PYCAMSET_CONFIG_DIR", None)
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, env=env,
                            check=True, capture_output=True, text=True, encoding="utf-8")
    data = json.loads(result.stdout.splitlines()[-1])
    resolved = Path(data["config"])
    assert resolved.is_absolute()
    assert not resolved.is_relative_to(repo)
    assert any(part.casefold() == "pycamset" for part in resolved.parts)
    assert data["application"] == "pyCamSet"
    assert data["organisation"] == "pyCamSet"
