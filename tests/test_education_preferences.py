"""Verify tolerant, separate education preference persistence.
"""
from __future__ import annotations

import json

import pytest


@pytest.mark.gui
def test_education_preferences_default_missing_and_unknown_fields_round_trip(tmp_path):
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_preferences import load_preferences, save_preferences

    path = tmp_path / "education_preferences.json"
    assert load_preferences(path) == {"shortcuts_enabled": True, "reduce_motion": False}
    path.write_text(json.dumps({
        "schema": "older-or-future", "version": 91,
        "preferences": {"reduce_motion": True, "future_option": "keep me"},
        "future_document_field": {"value": 7},
    }), encoding="utf-8")

    assert load_preferences(path) == {"shortcuts_enabled": True, "reduce_motion": True}
    save_preferences({"shortcuts_enabled": False}, path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["preferences"]["future_option"] == "keep me"
    assert saved["future_document_field"] == {"value": 7}
    assert saved["preferences"]["reduce_motion"] is True
    assert saved["preferences"]["shortcuts_enabled"] is False


@pytest.mark.gui
def test_education_preferences_bad_known_types_fall_back_but_are_not_rewritten(tmp_path):
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_preferences import load_preferences

    path = tmp_path / "education_preferences.json"
    original = '{"preferences":{"reduce_motion":1}}\n'
    path.write_text(original, encoding="utf-8")
    assert load_preferences(path) == {"shortcuts_enabled": True, "reduce_motion": False}
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.gui
def test_education_preferences_reject_non_boolean_updates(tmp_path):
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_preferences import save_preferences

    with pytest.raises(ValueError, match="boolean"):
        save_preferences({"reduce_motion": 1}, tmp_path / "education_preferences.json")
    with pytest.raises(ValueError, match="declared"):
        save_preferences({"metadata": {}}, tmp_path / "education_preferences.json")
