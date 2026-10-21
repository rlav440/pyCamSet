"""F1b wiring contracts: education entry points and the education-mode gate.

F1b's acceptance gate says "the flag hides everything".  The plan fixes the
*mechanism* -- "the flag is a persisted per-user preference" in
``education_preferences.json``, default off until the native acceptance check
passes -- but names no key, because no education key exists yet and F1a's
sidecar declares only ``shortcuts_enabled`` and ``reduce_motion``.  So the key
name and the helper are declared here, once, and pinned by a test.

Default off therefore also decides today's product behaviour: the four
Diagnostics entry buttons, the four run-finished entry points and the four
per-run recording checkboxes are hidden until the user turns the mode on in
that file (``education_preferences.json``, for testing).

Qt convention: this module gates itself with ``importorskip`` before its Qt
imports (the suite's module-top convention, tests/conftest.py:83-86).
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton


@pytest.fixture(scope="module")
def application():
    """Create the Qt application the offscreen widget contracts need."""
    return QApplication.instance() or QApplication([])


def _write_flag(path, enabled: bool) -> None:
    """Write the education sidecar with the mode flag set to *enabled*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "schema": "pycamset.education-preferences",
            "version": 1,
            "preferences": {"education_enabled": enabled},
        }),
        encoding="utf-8",
    )


def _run_tabs(window) -> dict:
    """The four phase run tabs, keyed by phase."""
    return {
        "phase1": window.phase1_tab,
        "phase2": window.phase2_tab,
        "phase3": window.phase3_tab,
        "phase4": window.phase4_tab,
    }


def _diagnostics_tabs(window) -> dict:
    """The four Diagnostics tabs, keyed by phase."""
    return {
        "phase1": window.phase1_diag_tab,
        "phase2": window.phase2_diag_tab,
        "phase3": window.phase3_diag_tab,
        "phase4": window.phase4_diag_tab,
    }


def _close(window, application) -> None:
    window.close()
    window.deleteLater()
    application.processEvents()


# ---------------------------------------------------------------------------
# The flag itself: location, key and default
# ---------------------------------------------------------------------------

def test_education_mode_is_off_when_the_sidecar_or_its_key_is_absent(tmp_path):
    from pyCamSet.gui.education_preferences import education_mode_enabled, load_preferences

    missing = tmp_path / "education_preferences.json"
    assert education_mode_enabled(missing) is False
    # The mode flag does not disturb F1a's presentation-key contract.
    assert load_preferences(missing) == {"shortcuts_enabled": True, "reduce_motion": False}

    _write_flag(missing, True)
    assert education_mode_enabled(missing) is True
    assert load_preferences(missing) == {"shortcuts_enabled": True, "reduce_motion": False}

    # A value that is not literally true is off, whatever JSON type it arrived as.
    _write_flag(missing, 1)
    assert education_mode_enabled(missing) is False
    missing.write_text("{ not json", encoding="utf-8")
    assert education_mode_enabled(missing) is False


def test_education_mode_flag_survives_a_save_that_does_not_mention_it(tmp_path):
    from pyCamSet.gui.education_preferences import education_mode_enabled, save_preferences

    path = tmp_path / "education_preferences.json"
    _write_flag(path, True)
    save_preferences({"reduce_motion": True}, path)
    assert education_mode_enabled(path) is True
    assert json.loads(path.read_text(encoding="utf-8"))["preferences"]["education_enabled"] is True


def test_education_mode_rejects_a_non_boolean_flag(tmp_path):
    from pyCamSet.gui.education_preferences import save_preferences

    with pytest.raises(ValueError):
        save_preferences({"education_enabled": 1}, tmp_path / "education_preferences.json")


# ---------------------------------------------------------------------------
# Default (off): the whole feature is hidden, and nothing records
# ---------------------------------------------------------------------------

def test_a_default_window_hides_every_education_entry_point(application, tmp_path, monkeypatch):
    """Absent sidecar -> off -> no button and no recording checkbox is visible."""
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    window.resize(1140, 820)
    window.show()
    application.processEvents()
    try:
        for phase, tab in _diagnostics_tabs(window).items():
            assert not tab._explain_btn.isVisibleTo(tab), f"{phase} entry button is visible"
        for phase, tab in _run_tabs(window).items():
            assert not tab._record_education_cb.isVisibleTo(tab), (
                f"{phase} recording checkbox is visible")
        assert not window.phase4_tab._explain_run_btn.isVisibleTo(window.phase4_tab)
    finally:
        _close(window, application)


def test_an_enabled_window_shows_the_entry_points_and_optin_controls(
        application, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    _write_flag(tmp_path / "education_preferences.json", True)
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    window.resize(1140, 820)
    window.show()
    application.processEvents()
    try:
        for phase, tab in _diagnostics_tabs(window).items():
            assert tab._explain_btn.isVisibleTo(tab), f"{phase} entry button is hidden"
        for phase, tab in _run_tabs(window).items():
            assert tab._record_education_cb.isVisibleTo(tab), (
                f"{phase} recording checkbox is hidden")
            assert tab._record_education_cb.isChecked() is False
        assert window.phase4_tab._explain_run_btn.isVisibleTo(window.phase4_tab)
    finally:
        _close(window, application)


def test_toggling_the_mode_hides_and_restores_the_controls(application, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    _write_flag(tmp_path / "education_preferences.json", True)
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    window.show()
    application.processEvents()
    try:
        tab = window.phase1_tab
        assert tab._record_education_cb.isVisibleTo(tab)
        window.set_education_enabled(False)
        application.processEvents()
        assert not tab._record_education_cb.isVisibleTo(tab)
        assert not window.phase1_diag_tab._explain_btn.isVisibleTo(window.phase1_diag_tab)
        window.set_education_enabled(True)
        application.processEvents()
        assert tab._record_education_cb.isVisibleTo(tab)
    finally:
        _close(window, application)


# ---------------------------------------------------------------------------
# The flag also turns recording off, whatever a widget still holds
# ---------------------------------------------------------------------------

def test_disabled_mode_forces_the_recording_parameter_off(application, tmp_path, monkeypatch):
    """A control hidden after a click must not still opt the run in.

    The checkbox is the only GUI route to the recording parameter; if the mode
    is switched off with the box ticked, the run must still record nothing.
    """
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    _write_flag(tmp_path / "education_preferences.json", True)
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    try:
        tab = window.phase1_tab
        tab.set_cameras(["camera-0"], selected_cameras=["camera-0"])
        tab._floc_edit.setText(str(tmp_path))
        tab._record_education_cb.setChecked(True)
        assert tab._read_params()["record_phase1_trace"] is True

        window.set_education_enabled(False)
        assert tab._read_params()["record_phase1_trace"] is False
        assert tab._record_education_cb.isChecked() is False
    finally:
        _close(window, application)


def test_the_recording_checkbox_reaches_the_runs_own_recording_key(application, tmp_path, monkeypatch):
    """Each phase's checkbox drives that producer's own opt-in key.

    Phase 1 and Phase 2 record under their producer cards' keys (B2's
    ``record_phase1_trace``, C1's ``record_phase2_per_view``); Phases 3 and 4
    share D1t's ``record_education``.
    """
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    _write_flag(tmp_path / "education_preferences.json", True)
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    try:
        window.phase1_tab.set_cameras(["camera-0"], selected_cameras=["camera-0"])
        window.phase1_tab._floc_edit.setText(str(tmp_path))
        window.phase1_tab._record_education_cb.setChecked(True)
        params = window.phase1_tab._read_params()
        assert params.get("record_phase1_trace") is True

        window.phase2_tab._record_education_cb.setChecked(True)
        assert window.phase2_tab._read_params().get("record_phase2_per_view") is True

        window.phase3_tab._record_education_cb.setChecked(True)
        assert window.phase3_tab._read_params()["record_education"] is True

        window.phase4_tab._record_education_cb.setChecked(True)
        assert window.phase4_tab._read_params()["record_education"] is True
    finally:
        _close(window, application)


# ---------------------------------------------------------------------------
# The run-finished entry point
# ---------------------------------------------------------------------------

def test_run_result_panel_withholds_the_entry_point_when_the_mode_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    from pyCamSet.gui import run_results

    run_dir = tmp_path / "phase1_runs" / "a-run"
    run_dir.mkdir(parents=True)
    panel = run_results.RunResultPanel("Detections", lambda: None)
    opened: list = []
    monkeypatch.setattr(run_results, "open_education_run",
                        lambda *args, **kwargs: opened.append(args))
    try:
        button = panel._explain_btn
        panel.set_education_run(run_dir)
        assert button.isVisibleTo(panel) is False
        assert button.isEnabled() is False
        button.click()
        assert opened == []

        _write_flag(tmp_path / "education_preferences.json", True)
        panel.set_education_run(run_dir)
        assert button.isEnabled() is True
    finally:
        panel.deleteLater()


def test_main_window_refresh_rereads_the_sidecar_and_updates_all_gates(
        application, tmp_path, monkeypatch):
    """Refreshing the persisted flag updates visibility and recording behaviour."""
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    _write_flag(tmp_path / "education_preferences.json", True)
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    window.show()
    application.processEvents()
    try:
        assert window.refresh_education_enabled() is True
        recording = window.phase1_tab._record_education_cb
        recording.setChecked(True)
        assert recording.isVisibleTo(window.phase1_tab) is True

        _write_flag(tmp_path / "education_preferences.json", False)
        assert window.refresh_education_enabled() is False
        assert recording.isVisibleTo(window.phase1_tab) is False
        assert recording.isChecked() is False
        assert not window.phase1_diag_tab._explain_btn.isVisibleTo(window.phase1_diag_tab)

        _write_flag(tmp_path / "education_preferences.json", True)
        assert window.refresh_education_enabled() is True
        assert recording.isVisibleTo(window.phase1_tab) is True
    finally:
        _close(window, application)


@pytest.mark.parametrize("phase,tab_name", [
    ("phase1", "phase1_tab"),
    ("phase2", "phase2_tab"),
    ("phase3", "phase3_tab"),
    ("phase4", "phase4_tab"),
])
@pytest.mark.parametrize("failed", [False, True], ids=["complete", "failed"])
def test_each_phase_finished_entry_opens_its_exact_run(
        application, tmp_path, monkeypatch, phase, tab_name, failed):
    """All four completion routes, including errors, open only their finished run."""
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    _write_flag(tmp_path / "education_preferences.json", True)
    from PySide6.QtWidgets import QLabel

    from pyCamSet.gui import run_results
    from pyCamSet.gui.main_window import PyCamSetApp

    run_dir = tmp_path / f"{phase}_runs" / "finished-run"
    run_dir.mkdir(parents=True)
    opened = []
    monkeypatch.setattr(
        run_results,
        "open_education_run",
        lambda run, selected_phase, workspace, parent: opened.append(
            (run, selected_phase, workspace)),
    )
    monkeypatch.setitem(
        run_results.VIEWS,
        phase,
        lambda run, workspace: (QLabel("result"), ""),
    )
    window = PyCamSetApp()
    window.show()
    application.processEvents()
    try:
        tab = getattr(window, tab_name)
        metadata = {
            "run_id": "finished-run",
            "status": "failed" if failed else "complete",
            "error": "solver failed" if failed else None,
            "_run_dir": str(run_dir),
        }
        tab._on_run_finished(metadata)

        button = (tab._explain_run_btn if phase == "phase4"
                  else tab._result_panel._explain_btn)
        assert button.isEnabled()
        button.click()
        if phase == "phase4":
            assert opened == [(metadata, "phase4", tab._workspace_mgr.workspace_path)]
        else:
            assert opened == [({"_run_dir": str(run_dir)}, phase, tmp_path)]

        window.set_education_enabled(False)
        assert not button.isVisibleTo(tab)
        button.click()
        assert len(opened) == 1
    finally:
        _close(window, application)


@pytest.mark.parametrize("tab_name,phase", [
    ("phase1_diag_tab", "phase1"),
    ("phase2_diag_tab", "phase2"),
    # Both Phase 3 and Phase 4 selectors merge runs from the two phases.
    ("phase3_diag_tab", "phase4"),
    ("phase4_diag_tab", "phase3"),
])
def test_diagnostics_entry_routes_one_selected_run_and_gates_invalid_selections(
        application, tmp_path, monkeypatch, tab_name, phase):
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    _write_flag(tmp_path / "education_preferences.json", True)

    from pyCamSet.gui import run_results
    from pyCamSet.gui.main_window import PyCamSetApp

    selected_run = {"run_id": "selected", "phase": phase}
    other_phase = "phase3" if phase == "phase4" else "phase4"
    other_run = {"run_id": "other", "phase": other_phase}
    opened = []
    monkeypatch.setattr(
        run_results,
        "open_education_run",
        lambda run, selected_phase, workspace, parent: opened.append(
            (run, selected_phase, workspace)),
    )
    window = PyCamSetApp()
    window.show()
    application.processEvents()
    try:
        tab = getattr(window, tab_name)
        tab.set_education_enabled(True)
        tab._run_selector.get_selected = lambda: [selected_run]
        tab._explain_btn.click()
        assert opened == [(selected_run, phase, tab._workspace_mgr.workspace_path)]

        tab._run_selector.get_selected = lambda: []
        tab._explain_btn.click()
        tab._run_selector.get_selected = lambda: [selected_run, other_run]
        tab._explain_btn.click()
        assert len(opened) == 1

        # The disabled button must remain inert even if click() is called directly.
        tab.set_education_enabled(False)
        tab._run_selector.get_selected = lambda: [selected_run]
        tab._explain_btn.click()
        assert len(opened) == 1
    finally:
        _close(window, application)
