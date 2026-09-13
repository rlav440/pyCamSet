"""The one thing the GUI remembers between sessions.

Everything about a calibration is already on disk under
``<image folder>/.pycamset_workspace``, and all of it is found again by
pointing the GUI at the same image folder.  The folder itself was the gap:
nothing recorded it, so a reopened GUI started blank with no way to
discover where it had been.

These cover the store, and the Phase 0 wiring that puts a remembered
folder back into the field -- from which the existing cross-tab
propagation repopulates every later phase on its own.
"""

from __future__ import annotations

import json

import pytest

from pyCamSet.workflow import recent_folders as rf
from pyCamSet.workflow import user_config as uc


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Never read or write the developer's own settings."""
    config = tmp_path / "config"
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(config))
    return config


def _folder(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


# --------------------------------------------------------------------------
# Where it lives
# --------------------------------------------------------------------------


def test_the_override_decides_the_location(isolated_config):
    assert uc.config_dir() == isolated_config
    assert rf.recent_folders_file() == isolated_config / "recent_folders.json"


def test_each_platform_has_a_home_for_it(monkeypatch):
    """The location is shared with every other remembered list, so it is
    asked of the settings module rather than of this one."""
    monkeypatch.delenv("PYCAMSET_CONFIG_DIR", raising=False)

    monkeypatch.setattr(uc.sys, "platform", "darwin")
    assert uc.config_dir().parts[-3:] == ("Library", "Application Support", "pyCamSet")

    # sys.platform only.  uc.os is the os module itself, so patching its
    # name would set os.name for the whole process, and on a Windows host
    # pathlib then builds PosixPath and raises -- including inside pytest's
    # own failure reporting, which turns any later failure into an
    # INTERNALERROR that takes the session down.
    monkeypatch.setattr(uc.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg")
    assert uc.config_dir().as_posix() == "/tmp/xdg/pyCamSet"


# --------------------------------------------------------------------------
# The list
# --------------------------------------------------------------------------


def test_nothing_remembered_yet_is_an_empty_list():
    assert rf.load_recent_folders() == []


def test_a_folder_comes_back(tmp_path):
    folder = _folder(tmp_path, "project")

    rf.remember_folder(folder)

    assert rf.load_recent_folders() == [folder.resolve()]


def test_the_most_recent_is_first(tmp_path):
    first = _folder(tmp_path, "first")
    second = _folder(tmp_path, "second")

    rf.remember_folder(first)
    rf.remember_folder(second)

    assert rf.load_recent_folders() == [second.resolve(), first.resolve()]


def test_remembering_again_moves_it_up_rather_than_duplicating(tmp_path):
    first = _folder(tmp_path, "first")
    second = _folder(tmp_path, "second")

    rf.remember_folder(first)
    rf.remember_folder(second)
    rf.remember_folder(first)

    assert rf.load_recent_folders() == [first.resolve(), second.resolve()]


def test_the_list_stays_a_list(tmp_path):
    """It is somewhere to pick from, not a history."""
    for i in range(rf.MAX_RECENT + 5):
        rf.remember_folder(_folder(tmp_path, f"project_{i:02d}"))

    remembered = rf.load_recent_folders()

    assert len(remembered) == rf.MAX_RECENT
    assert remembered[0].name == f"project_{rf.MAX_RECENT + 4:02d}"


def test_folders_that_are_gone_are_not_offered(tmp_path):
    """Removable drives and deleted projects otherwise accumulate."""
    kept = _folder(tmp_path, "kept")
    removed = _folder(tmp_path, "removed")
    rf.remember_folder(kept)
    rf.remember_folder(removed)
    removed.rmdir()

    assert rf.load_recent_folders() == [kept.resolve()]
    # still on disk, in case the drive comes back
    assert len(rf.load_recent_folders(existing_only=False)) == 2


def test_a_folder_can_be_forgotten(tmp_path):
    kept = _folder(tmp_path, "kept")
    dropped = _folder(tmp_path, "dropped")
    rf.remember_folder(kept)
    rf.remember_folder(dropped)

    rf.forget_folder(dropped)

    assert rf.load_recent_folders() == [kept.resolve()]


def test_remembering_nothing_does_nothing():
    rf.remember_folder("")
    assert rf.load_recent_folders() == []


# --------------------------------------------------------------------------
# It must never be the reason a launch fails
# --------------------------------------------------------------------------


def test_an_unreadable_file_reads_as_empty(isolated_config):
    isolated_config.mkdir(parents=True)
    (isolated_config / "recent_folders.json").write_text("{ this is not json")

    assert rf.load_recent_folders() == []


def test_a_file_of_the_wrong_shape_reads_as_empty(isolated_config):
    isolated_config.mkdir(parents=True)
    (isolated_config / "recent_folders.json").write_text(json.dumps({"folders": "nope"}))

    assert rf.load_recent_folders() == []


def test_a_config_directory_that_cannot_be_written_is_survivable(
        tmp_path, monkeypatch):
    """A read-only home must cost the memory, not the GUI."""
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path / "config"))
    project = _folder(tmp_path, "project")   # before the filesystem "fails"

    def _refuse(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(rf.Path, "mkdir", _refuse)

    rf.remember_folder(project)   # must not raise
    assert rf.load_recent_folders() == []


# --------------------------------------------------------------------------
# What records a folder
# --------------------------------------------------------------------------


def test_saving_a_run_records_nothing(tmp_path):
    """The regression: temporary folders in the list of places to go back to.

    Runs are written by the tests, by a parameter search and by any script
    that calls a phase, nearly always into a temporary directory.  Recording
    them from the save put a trail of ``/tmp/tmp8f3k/images`` in front of the
    person who had never seen any of them.
    """
    from pyCamSet.workflow.workspace import WorkspaceManager

    image_folder = _folder(tmp_path, "images")
    manager = WorkspaceManager(image_folder / ".pycamset_workspace")

    manager.save_run("phase1", "run_a", {"phase": "phase1"})

    assert rf.load_recent_folders() == []


@pytest.mark.gui
def test_running_a_phase_records_the_folder_it_ran_on(tmp_path, monkeypatch):
    """Where someone works is something only the interface knows: a folder
    they set and then ran a phase on is one they will want offered again."""
    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.gui import shared_functions as shared
    from pyCamSet.gui.phase_1_detection import Phase1Tab
    from pyCamSet.workflow import phase1
    from pyCamSet.workflow.workspace import WorkspaceManager

    images = tmp_path / "images"
    for camera in ("cam0", "cam1"):
        (images / camera).mkdir(parents=True)
    monkeypatch.setattr(phase1, "run", lambda *a, **k: {"run_id": "x"})
    monkeypatch.setattr(shared.PhaseWorker, "start",
                        lambda self: self._work_fn(lambda _line: None))

    QApplication.instance() or QApplication([])
    tab = Phase1Tab(notebook=QTabWidget(), info_cb=QCheckBox(),
                    terminal_cb=QCheckBox(), workspace_mgr=WorkspaceManager(None))
    try:
        tab._floc_edit.setText(str(images))
        tab.set_cameras(["cam0", "cam1"], ["cam0", "cam1"])
        tab._run_phase1()
    finally:
        tab.deleteLater()

    assert rf.load_recent_folders() == [images.resolve()]


# --------------------------------------------------------------------------
# Phase 0, and the phases that follow from it
# --------------------------------------------------------------------------


@pytest.fixture
def qt_app():
    pytest.importorskip("PySide6", reason="the phase tabs are Qt widgets")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def calibration_folder(tmp_path):
    """An image folder shaped the way Phase 0 requires."""
    folder = tmp_path / "rig"
    for cam in ("cam0", "cam1"):
        (folder / cam).mkdir(parents=True)
        for i in range(2):
            (folder / cam / f"{i}.png").write_bytes(b"")
    return folder


@pytest.mark.gui
def test_a_fresh_launch_offers_nothing_and_touches_nothing(qt_app):
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()

    combo = window.phase0_tab._recent_combo
    assert combo.count() == 1
    assert combo.itemData(0) is None
    # the lazy workspace is deliberate: launching must not create anything
    assert window._workspace_mgr.workspace_path is None


@pytest.mark.gui
def test_a_confirmed_folder_is_offered_next_time(qt_app, calibration_folder):
    from pyCamSet.gui.main_window import PyCamSetApp

    first = PyCamSetApp()
    first.phase0_tab._floc_edit.setText(str(calibration_folder))
    first.phase0_tab._confirm_image_folder_validity()

    second = PyCamSetApp()
    offered = [second.phase0_tab._recent_combo.itemData(i)
               for i in range(second.phase0_tab._recent_combo.count())]

    assert str(calibration_folder.resolve()) in offered


@pytest.mark.gui
def test_choosing_a_recent_folder_fills_in_the_later_phases(
        qt_app, calibration_folder):
    """The point of remembering it: one click restores the session.

    Nothing here populates the phases directly -- setting the folder does,
    because every phase tab already watches that field.
    """
    from pyCamSet.gui.main_window import PyCamSetApp

    rf.remember_folder(calibration_folder)
    window = PyCamSetApp()
    combo = window.phase0_tab._recent_combo
    index = next(i for i in range(combo.count())
                 if combo.itemData(i) == str(calibration_folder.resolve()))

    combo.setCurrentIndex(index)
    window.phase0_tab._on_recent_selected(index)

    assert window._workspace_mgr.workspace_path == (
        calibration_folder.resolve() / ".pycamset_workspace")
    for tab in ("phase1_tab", "phase2_tab", "phase3_tab", "phase4_tab"):
        assert getattr(window, tab)._floc_edit.text()


@pytest.mark.gui
def test_a_folder_that_moved_is_reported_without_a_dialog(
        qt_app, calibration_folder, monkeypatch):
    """Mid-selection is the wrong moment for a modal."""
    from PySide6.QtWidgets import QMessageBox

    from pyCamSet.gui.main_window import PyCamSetApp

    rf.remember_folder(calibration_folder)
    window = PyCamSetApp()

    def _no_dialogs(*args, **kwargs):
        raise AssertionError("a dialog was raised while picking a folder")

    for name in ("critical", "warning", "information"):
        monkeypatch.setattr(QMessageBox, name, _no_dialogs)

    window.phase0_tab._floc_edit.setText(str(calibration_folder / "gone"))
    window.phase0_tab._confirm_image_folder_validity(quiet=True)

    assert "no longer there" in window.phase0_tab._status_lbl.text()


@pytest.mark.gui
def test_forgetting_a_folder_removes_it_from_the_list(
        qt_app, calibration_folder):
    from pyCamSet.gui.main_window import PyCamSetApp

    rf.remember_folder(calibration_folder)
    window = PyCamSetApp()
    combo = window.phase0_tab._recent_combo
    index = next(i for i in range(combo.count())
                 if combo.itemData(i) == str(calibration_folder.resolve()))
    combo.setCurrentIndex(index)

    window.phase0_tab._forget_selected_recent()

    assert rf.load_recent_folders() == []
    assert combo.count() == 1
    assert combo.itemData(0) is None
