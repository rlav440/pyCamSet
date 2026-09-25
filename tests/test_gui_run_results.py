"""The result card beside each phase's controls."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_best_and_worst_detections_skip_images_without_any():
    from pyCamSet.gui.run_results import detection_extremes

    features = np.array([[24, 0, 7],
                         [33, 3, 0],
                         [0, 12, 40]])
    best, worst, empty = detection_extremes(features)
    assert best == (2, 2)          # image 2, camera 2: 40 detections
    assert worst == (1, 1)         # the fewest that are not none
    assert empty == 3
    assert detection_extremes(np.zeros((2, 2))) is None


def test_phase2_figure_shows_each_image_and_each_camera(qapp):
    from pyCamSet.gui.run_results import phase2_figure

    run = {"diagnostics": {
        "D2.1_per_camera_rms_reprojection": {"camA": 1.2, "camB": 3.4},
        "D2.6_per_view_reprojection": {
            "camA": {"rms_px": [1.0, 1.1, float("nan")]},
            "camB": {"rms_px": [2.0, 9.0]},
        }}}
    figure = phase2_figure(run)
    axes = figure.axes[0]
    offsets = [collection.get_offsets() for collection in axes.collections]
    assert sum(len(o) for o in offsets) == 2 + 2 + 2  # finite images + two RMS marks
    assert [t.get_text() for t in axes.get_xticklabels()] == ["camA", "camB"]
    assert phase2_figure({"diagnostics": {}}) is None


def _cam(name, position, view):
    return SimpleNamespace(name=name, position=np.array(position, float), view=np.array(view, float))


def test_pinhole_cameras_are_drawn_where_they_are():
    from pyCamSet.gui.run_results import camera_layout

    cams = [_cam("a", [1, 0, 0], [-1, 0, 0]), _cam("b", [0, 2, 0], [0, -1, 0])]
    positions, views, names, placed = camera_layout(cams)
    assert not placed
    np.testing.assert_allclose(positions, [[1, 0, 0], [0, 2, 0]])
    assert names == ["a", "b"]


def test_telecentric_cameras_are_placed_back_along_their_view():
    from pyCamSet.gui.run_results import camera_layout

    cams = [_cam("a", [0, 0, 0], [0, 0, 2]), _cam("b", [0, 0, 0], [1, 0, 0])]
    positions, views, _names, placed = camera_layout(cams)
    assert placed
    np.testing.assert_allclose(positions, [[0, 0, -1], [-1, 0, 0]])
    np.testing.assert_allclose(np.linalg.norm(views, axis=1), 1.0)


def test_pose_labels_carry_each_cameras_error(qapp):
    from pyCamSet.gui.run_results import phase3_figure

    cams = [_cam("left", [1, 0, 0], [-1, 0, 0]), _cam("right", [-1, 0, 0], [1, 0, 0])]
    figure = phase3_figure(cams, {"left": 1.234, "right": 5.0})
    texts = [t.get_text() for t in figure.axes[0].texts]
    assert "left\n1.23 px" in texts and "right\n5.00 px" in texts


def test_card_hides_for_a_failed_run_and_reports_a_broken_one(qapp, monkeypatch):
    from pyCamSet.gui import run_results

    panel = run_results.RunResultPanel("Detections", lambda: None)
    run_results.show_run_result(panel, "phase1", {"error": "boom"}, None)
    assert panel.isHidden()

    def broken(run, workspace):
        raise OSError("disk gone")

    monkeypatch.setitem(run_results.VIEWS, "phase1", broken)
    run_results.show_run_result(panel, "phase1", {"run_id": "r"}, None)
    assert not panel.isHidden()
    assert "disk gone" in panel.message.text()


def test_see_more_opens_diagnostics(qapp):
    from pyCamSet.gui.run_results import RunResultPanel

    opened = []
    panel = RunResultPanel("Intrinsics", lambda: opened.append(True))
    assert panel.more.text() == "See more in Diagnostics"
    panel.more.click()
    assert opened == [True]


@pytest.mark.parametrize("tab_name,phase", [
    ("phase1_tab", "phase1"), ("phase2_tab", "phase2"), ("phase3_tab", "phase3")])
def test_each_phase_fills_its_card_when_a_run_finishes(qapp, tmp_path, monkeypatch, tab_name, phase):
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    from PySide6.QtWidgets import QLabel

    from pyCamSet.gui import run_results
    from pyCamSet.gui.main_window import PyCamSetApp

    seen = []
    monkeypatch.setitem(run_results.VIEWS, phase,
                        lambda run, ws: seen.append(run["run_id"]) or (QLabel("drawn"), ""))
    window = PyCamSetApp()
    try:
        tab = getattr(window, tab_name)
        assert tab._result_panel.isHidden()
        tab._on_run_finished({"run_id": "r1", "status": "complete"})
        assert seen == ["r1"]
        assert not tab._result_panel.isHidden()
        assert tab._side.isHidden()  # the card takes the empty column's place
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
