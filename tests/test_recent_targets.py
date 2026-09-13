"""The other thing the GUI remembers between sessions.

A target is a handful of numbers that must match the printed object exactly,
and nothing in a workspace records them until a run has already been made
with them -- so they were retyped from memory every session, and a wrong one
shows up only as detections that come back empty.

These cover the store, and the Phase 1 wiring that puts a remembered target
back into the form.
"""

from __future__ import annotations

import json

import pytest

from pyCamSet.workflow import recent_targets as rt
from pyCamSet.workflow import user_config as uc

CCUBE = {"type": "Ccube", "n_points": 10, "length": 40.0,
         "border_fraction": 0.2, "marker_backend": "aruco1"}
CHARUCO = {"type": "ChArUco", "num_squares_x": 20, "num_squares_y": 20,
           "square_size": 4.0, "marker_fraction": 0.8}


def _stored() -> list[dict]:
    """The list as it sits on disk."""
    with open(uc.config_dir() / "recent_targets.json", encoding="utf-8") as fh:
        return json.load(fh)["targets"]


# --------------------------------------------------------------------------
# The list
# --------------------------------------------------------------------------

def test_nothing_is_remembered_to_begin_with():
    assert rt.load_recent_targets() == []


def test_a_remembered_target_comes_back_whole():
    rt.remember_target(CCUBE)

    assert rt.load_recent_targets() == [CCUBE]
    assert _stored() == [CCUBE]


def test_the_most_recent_target_is_first():
    rt.remember_target(CCUBE)
    rt.remember_target(CHARUCO)

    assert [t["type"] for t in rt.load_recent_targets()] == ["ChArUco", "Ccube"]


def test_the_same_target_moves_rather_than_repeats():
    rt.remember_target(CCUBE)
    rt.remember_target(CHARUCO)
    rt.remember_target(CCUBE)

    assert [t["type"] for t in rt.load_recent_targets()] == ["Ccube", "ChArUco"]


def test_a_board_of_a_different_size_is_a_different_target():
    rt.remember_target(CHARUCO)
    rt.remember_target({**CHARUCO, "num_squares_x": 10})

    assert len(rt.load_recent_targets()) == 2


def test_retuning_the_detector_does_not_make_a_new_target():
    """The detector options say how a marker is read, not what the target is:
    remembering each tuning would fill the list with the same board."""
    rt.remember_target({**CCUBE, "detection_options": {"adaptiveThreshWinSizeMin": 3}})
    rt.remember_target({**CCUBE, "detection_options": {"adaptiveThreshWinSizeMin": 5}})

    remembered = rt.load_recent_targets()
    assert len(remembered) == 1
    assert remembered[0]["detection_options"] == {"adaptiveThreshWinSizeMin": 5}


def test_the_list_stays_short():
    for size in range(uc.MAX_RECENT + 5):
        rt.remember_target({**CHARUCO, "num_squares_x": size})

    remembered = rt.load_recent_targets()
    assert len(remembered) == uc.MAX_RECENT
    assert remembered[0]["num_squares_x"] == uc.MAX_RECENT + 4


def test_a_target_can_be_forgotten():
    rt.remember_target(CCUBE)
    rt.remember_target(CHARUCO)

    rt.forget_target(CCUBE)

    assert [t["type"] for t in rt.load_recent_targets()] == ["ChArUco"]


def test_nothing_without_a_type_is_remembered():
    rt.remember_target({})
    rt.remember_target({"n_points": 10})

    assert rt.load_recent_targets() == []


def test_a_damaged_list_is_not_worth_failing_a_launch_over():
    path = uc.config_dir() / "recent_targets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")

    assert rt.load_recent_targets() == []


def test_a_config_directory_that_cannot_be_written_is_survivable(monkeypatch):
    monkeypatch.setattr(
        uc.Path, "mkdir",
        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))

    rt.remember_target(CCUBE)          # says nothing, raises nothing

    assert rt.load_recent_targets() == []


# --------------------------------------------------------------------------
# The Phase 1 wiring
# --------------------------------------------------------------------------

@pytest.fixture
def phase1_tab():
    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.gui.phase_1_detection import Phase1Tab
    from pyCamSet.gui.shared_functions import WorkspaceManager

    QApplication.instance() or QApplication([])
    tab = Phase1Tab(notebook=QTabWidget(), info_cb=QCheckBox(),
                    terminal_cb=QCheckBox(), workspace_mgr=WorkspaceManager(None))
    yield tab
    tab.deleteLater()


@pytest.mark.gui
def test_picking_a_remembered_target_fills_the_form_in(phase1_tab):
    rt.remember_target(CHARUCO)
    phase1_tab.refresh_recent_targets()

    # Index 0 is the prompt, so the first remembered target is index 1.
    phase1_tab._recent_target_combo.setCurrentIndex(1)
    phase1_tab._on_recent_target_selected(1)

    assert phase1_tab._target_combo.currentText() == "ChArUco"
    assert phase1_tab._npts_spin.value() == 20
    assert phase1_tab._length_edit.text() == "4"


@pytest.mark.gui
def test_the_combo_names_each_target_by_what_it_is(phase1_tab):
    rt.remember_target(CCUBE)
    phase1_tab.refresh_recent_targets()

    assert "Ccube" in phase1_tab._recent_target_combo.itemText(1)
    assert "n_points=10" in phase1_tab._recent_target_combo.itemText(1)


@pytest.mark.gui
def test_running_phase_1_remembers_the_target_it_ran_with(phase1_tab, tmp_path,
                                                          monkeypatch):
    """Remembered where the target is known to be one a phase will run with,
    the same rule the image folders follow."""
    from pyCamSet.gui import shared_functions as shared
    from pyCamSet.workflow import phase1

    images = tmp_path / "images"
    for camera in ("cam0", "cam1"):
        (images / camera).mkdir(parents=True)
    monkeypatch.setattr(phase1, "run", lambda *a, **k: {"run_id": "x"})
    monkeypatch.setattr(shared.PhaseWorker, "start",
                        lambda self: self._work_fn(lambda _line: None))

    phase1_tab._floc_edit.setText(str(images))
    phase1_tab.set_cameras(["cam0", "cam1"], ["cam0", "cam1"])
    phase1_tab._target_combo.setCurrentText("ChArUco")
    phase1_tab._npts_spin.setValue(20)
    phase1_tab._length_edit.setText("4")

    phase1_tab._run_phase1()

    remembered = rt.load_recent_targets()
    assert remembered and remembered[0]["type"] == "ChArUco"
    assert remembered[0]["num_squares_x"] == 20
