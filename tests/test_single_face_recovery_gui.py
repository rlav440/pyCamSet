"""Phase 3 GUI for the optional single-face telecentric recovery: the setting and its D3.14 section.

On a real telecentric rig, 17 of 99 images were left out of Phase 3 because every
telecentric camera saw only one cube face of each. The recovery is on in the library and ticked
by default in the GUI; the Diagnostics tab lists what it placed and what it left out.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QTabWidget  # noqa: E402


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


_ON = {
    "enabled": True, "telecentric": True,
    "recovered": [{"image": 12, "rms_px": 2.345, "cameras": ["cam0", "cam3"]}],
    "not_recovered": [
        {"image": 40, "reason": "seen by one camera only: the tilt cannot be resolved", "cameras": ["cam5"]},
        {"image": 41, "reason": "best fit 33.10 px exceeds the 20 px limit", "cameras": ["cam1", "cam2"]},
    ],
    "teaching": {},
}
_OFF = {
    "enabled": False, "telecentric": True, "recovered": [], "teaching": {},
    "not_recovered": [{"image": 7, "cameras": ["cam0", "cam1"],
                       "reason": "recovery off: every camera saw one face (recoverable in principle)"}],
}


def test_the_checkbox_is_ticked_by_default_and_reaches_the_phase3_params(application, tmp_path):
    from pyCamSet.gui.phase_3_bundle_adjustment import Phase3Tab
    from pyCamSet.workflow.workspace import WorkspaceManager

    tab = Phase3Tab(QTabWidget(), QCheckBox(), QCheckBox(), WorkspaceManager())
    try:
        box = tab._single_face_cb
        assert box.isChecked()
        assert box.text() == "Recover images where each camera sees only one cube face (telecentric)"
        assert "no effect on pinhole cameras" in box.toolTip()
        tab._floc_edit.setText(str(tmp_path))
        assert tab._read_params()["recover_single_face_telecentric"] is True
        box.setChecked(False)
        assert tab._read_params()["recover_single_face_telecentric"] is False
    finally:
        tab.deleteLater()
        application.processEvents()


def test_summary_lines_list_recovered_and_left_out_images_and_the_off_state():
    from pyCamSet.gui.phase_3_bundle_adjustment import single_face_summary_lines

    on = single_face_summary_lines(_ON)
    assert on[0] == "On: 1 recovered, 2 left out."
    assert "Recovered: image 12, fit 2.35 px (cam0, cam3)" in on
    assert ("Left out: image 40, seen by one camera only: the tilt cannot be resolved (cam5)") in on
    assert "Left out: image 41, best fit 33.10 px exceeds the 20 px limit (cam1, cam2)" in on
    off = single_face_summary_lines(_OFF)
    assert off[0] == "Recovery off for this run."
    assert off[1].startswith("Left out: image 7, recovery off")
    assert single_face_summary_lines({**_ON, "telecentric": False, "recovered": [], "not_recovered": []}) == [
        "On, no effect: the cameras are not telecentric."]
    assert single_face_summary_lines({**_ON, "recovered": [], "not_recovered": []})[-1] == "No image needed it."


@pytest.mark.parametrize("record, expected", [(_ON, "Recovered: image 12"), (_OFF, "Recovery off")])
def test_the_diagnostics_summary_renders_the_section(application, record, expected):
    from pyCamSet.gui.phase_3_bundle_adjustment import Phase3DiagnosticsTab
    from pyCamSet.workflow.workspace import WorkspaceManager

    tab = Phase3DiagnosticsTab(QTabWidget(), QCheckBox(), WorkspaceManager())
    def sections():
        return [label for label in tab._summary_inner.findChildren(QLabel)
                if label.objectName() == "single_face_summary"]

    try:
        # An older run without D3.14 shows no section.
        tab._render_summary([{"run_id": "r0", "phase": "phase3", "status": "complete", "diagnostics": {}}])
        assert sections() == []
        tab._render_summary([{"run_id": "r1", "phase": "phase3", "status": "complete",
                              "diagnostics": {"D3.1_n_missing_poses": 2,
                                              "D3.14_single_face_recovery": record}}])
        [label] = sections()
        assert expected in label.text()
        assert label.accessibleName() == "Single-face images (D3.14)"
    finally:
        tab.deleteLater()
        application.processEvents()
